import os
import argparse
import inspect
import torch
import torch.nn.functional as F
import json
from torch_geometric.data import Dataset, Data
import torch_geometric.datasets as pyg_datasets
import torch_geometric.nn as pyg_nn
from loguru import logger
from mcp.server.fastmcp import FastMCP

import openai

# ---- Logging ----
def configure_logger(log_file: str = None) -> None:
    if log_file is None:
        log_file = os.path.abspath("debug.log")
    logger.remove()  # remove default logger to stderr
    if os.path.exists(log_file):
        os.remove(log_file)
    logger.add(log_file, format="{time} {level} {message}", level="DEBUG")


device = torch.device(
    'cuda:1' if torch.cuda.is_available() and torch.cuda.device_count() > 1
    else 'cuda:0' if torch.cuda.is_available()
    else 'cpu'
)
logger.info(f"[BOOT] Using device: {device}")

DATA_ROOT = os.path.abspath("data")
KNOWN_DATASETS = {
    'cora': ('Planetoid', 'Cora'),
    'citeseer': ('Planetoid', 'Citeseer'),
    'pubmed': ('Planetoid', 'Pubmed'),
    'cornell': ('WebKB', 'Cornell'),
    'texas': ('WebKB', 'Texas'),
    'wisconsin': ('WebKB', 'Wisconsin'),
    'actor': ('Actor', None),
}

def get_dataset(name: str) -> Dataset:
    logger.info(f"[DATA] Getting dataset: {name}")
    key = name.lower()
    if key in KNOWN_DATASETS:
        cls, arg = KNOWN_DATASETS[key]
        mod = getattr(pyg_datasets, cls)
        params = {'root': os.path.join(DATA_ROOT, cls)}
        if arg:
            params['name'] = arg
        ds = mod(**params)
        logger.info(f"[DATA] Loaded dataset '{name}' (class: {cls}, param: {arg})")
        return ds
    for nm, cls in inspect.getmembers(pyg_datasets, inspect.isclass):
        if issubclass(cls, Dataset) and nm.lower() == key:
            params = {'root': os.path.join(DATA_ROOT, nm)}
            sig = inspect.signature(cls)
            if 'name' in sig.parameters:
                params['name'] = name
            ds = cls(**params)
            logger.info(f"[DATA] Loaded fallback dataset '{name}' (class: {nm})")
            return ds
    logger.error(f"[DATA] Unknown dataset: {name}")
    raise ValueError(f"Unknown dataset: {name}")

def get_dataset_info(dataset: Dataset) -> dict:
    logger.info(f"[DATA] Gathering info for dataset")
    data = dataset[0]
    info = {
        'num_node_features': dataset.num_node_features,
        'num_classes': dataset.num_classes,
        'num_nodes': int(data.num_nodes),
    }
    logger.info(f"[DATA] Info: {info}")
    return info

def clean_layers(layers):
    """Remove Dropout from layers (must be handled as a param, not layer)."""
    return [l for l in layers if l.get('type', '').lower() != 'dropout']

class GNN(torch.nn.Module):
    def __init__(self, layers_config: list, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        logger.info(f"[GNN] Initializing model, layers: {layers_config}, in: {in_channels}, out: {out_channels}")
        self.layers = torch.nn.ModuleList()
        self.dropout = dropout
        current_ch = in_channels
        for idx, cfg in enumerate(layers_config):
            if cfg.get('type', '').lower() == "dropout":
                logger.warning(f"[GNN] Ignoring Dropout layer in layers config; use 'dropout' as parameter.")
                continue
            layer_cls = getattr(pyg_nn, cfg['type'], None)
            if layer_cls is None:
                logger.error(f"[GNN] Unknown layer type: {cfg['type']}")
                raise ValueError(f"Unknown layer type: {cfg['type']}")
            is_last = (idx == len(layers_config) - 1)
            o_ch = out_channels if is_last else cfg.get('out_channels', current_ch)
            params = {'in_channels': current_ch, 'out_channels': o_ch}
            params.update(cfg.get('additional_params', {}))
            logger.info(f"[GNN] Adding layer {cfg['type']}({current_ch}, {o_ch})")
            self.layers.append(layer_cls(**params))
            current_ch = o_ch
        act_name = layers_config[0].get('activation', 'relu')
        self.activation = getattr(F, act_name)
    def forward(self, data: Data):
        x, edge_index = data.x, data.edge_index
        for layer in self.layers:
            x = layer(x, edge_index)
            x = self.activation(x)
            if self.dropout > 0:
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x
    def structure(self) -> dict:
        return {'layers': [type(l).__name__ for l in self.layers]}
    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

def train_epoch(model: GNN, optimizer: torch.optim.Optimizer, data: Data) -> float:
    model.train()
    optimizer.zero_grad()
    out = model(data)
    mask = data.train_mask
    loss = F.nll_loss(F.log_softmax(out, dim=1)[mask], data.y[mask])
    loss.backward()
    optimizer.step()
    logger.info(f"[TRAIN] Loss: {loss.item():.4f}")
    return float(loss.item())

def evaluate(model: GNN, data: Data) -> dict:
    model.eval()
    out = model(data)
    preds = out.argmax(dim=1)
    results = {}
    for split in ['train', 'test']:
        mask = getattr(data, f"{split}_mask")
        correct = (preds[mask] == data.y[mask]).sum().item()
        total = mask.sum().item()
        acc = correct / total if total > 0 else 0.0
        results[f"{split}_accuracy"] = acc
        logger.info(f"[EVAL] {split}_accuracy: {acc:.4f}")
    return results

# --------- MCP TOOLS ---------
mcp = FastMCP('gnn-designer')
STATE: dict = {}

@mcp.tool()
def dataset_info(dataset_name: str) -> dict:
    logger.info(f"[TOOL] dataset_info({dataset_name})")
    ds = get_dataset(dataset_name)
    info = get_dataset_info(ds)
    STATE['last_dataset'] = ds
    STATE['last_info'] = info
    return info

@mcp.tool()
def run_iteration(dataset_name: str, hyperparams: dict, epochs: int = 1) -> dict:
    logger.info(f"[TOOL] run_iteration({dataset_name}, {hyperparams}, epochs={epochs})")
    ds = get_dataset(dataset_name)
    data = ds[0].to(device)
    # clean Dropout from layers, use dropout as separate param
    layers = clean_layers(hyperparams['layers'])
    dropout = hyperparams.get('dropout', 0.0)
    model = GNN(layers, ds.num_node_features, ds.num_classes, dropout=dropout).to(device)
    if hyperparams.get('weight_initialization') == 'xavier':
        for module in model.modules():
            if hasattr(module, 'weight'):
                torch.nn.init.xavier_uniform_(module.weight)
    OptimClass = getattr(torch.optim, hyperparams.get('optimizer', 'Adam'))
    optimizer = OptimClass(
        model.parameters(),
        lr=hyperparams['lr'],
        weight_decay=hyperparams['weight_decay']
    )
    for epoch in range(epochs):
        logger.info(f"[TRAIN] Epoch {epoch+1}/{epochs}")
        train_epoch(model, optimizer, data)
    metrics = evaluate(model, data)
    struct = model.structure()
    result = {
        'metrics': metrics,
        'structure': struct,
        'num_parameters': model.param_count()
    }
    logger.info(f"[TOOL] run_iteration result: {result}")
    STATE['last_params'] = hyperparams
    STATE['last_struct'] = struct
    STATE['last_metrics'] = metrics
    return result

@mcp.tool()
def get_params() -> dict:
    logger.info(f"[TOOL] get_params()")
    return STATE.get('last_params', {})

@mcp.tool()
def get_hyperparams() -> dict:
    logger.info(f"[TOOL] get_hyperparams()")
    return STATE.get('best_params', {})

@mcp.tool()
def get_best_architecture() -> dict:
    logger.info(f"[TOOL] get_best_architecture()")
    return STATE.get('last_struct', {})

@mcp.tool()
def get_best_results() -> dict:
    logger.info(f"[TOOL] get_best_results()")
    return STATE.get('best_metrics', {})

@mcp.tool()
def get_best_hyperparams() -> dict:
    logger.info(f"[TOOL] get_best_hyperparams()")
    return STATE.get('best_params', {})

@mcp.tool()
def train_best(dataset_name: str, epochs: int = 1) -> dict:
    logger.info(f"[TOOL] train_best({dataset_name}, epochs={epochs})")
    best = STATE.get('best_params')
    if not best:
        logger.error("[TOOL] No best parameters available.")
        return {'error': 'No best parameters available.'}
    return run_iteration(dataset_name, best, epochs)

# ---------- LLM tools: Ollama llama3.2 ----------
def call_llm(prompt: str, llm_model: str = "llama3.2", max_retries: int = 5) -> dict:
    for attempt in range(max_retries):
        try:
            client = openai.OpenAI(
                base_url="http://localhost:11434/v1",  # Ollama REST endpoint
                api_key="ollama"
            )
            response = client.chat.completions.create(
                model=llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You only reply with a valid, compact JSON dict for Python, never markdown or code blocks. "
                            "Do NOT include Dropout as a layer in the layers list. "
                            "If Dropout should be used, set it as a top-level key (e.g., \"dropout\": 0.5)."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ]
            )
            reply = response.choices[0].message.content.strip()
            logger.info(f"[LLM] Ollama reply: {reply}")
            # Usuń otaczające code blocki, jeśli są
            if reply.startswith("```"):
                reply = reply.strip("`").strip()
                if reply.startswith("json"):
                    reply = reply[4:].strip()
            return json.loads(reply)
        except Exception as ex:
            logger.error(f"[LLM] Could not parse LLM reply (attempt {attempt+1}): {ex}")
    return {"error": f"Could not parse LLM reply after {max_retries} tries."}

@mcp.tool()
def suggest_hyperparams(
    llm_model: str,
    current_metrics: dict,
    current_params: dict,
    last_structure: dict,
    best_structure: dict,
    dataset_name: str
) -> dict:
    logger.info(f"[TOOL] suggest_hyperparams({llm_model}, {current_metrics}, ...dataset={dataset_name})")
    ds = get_dataset(dataset_name)
    ds_info = get_dataset_info(ds)
    prompt = f"""
You are an expert in neural network hyperparameter tuning. The following graph dataset info is provided:
{json.dumps(ds_info, indent=2)}
Current metrics: {json.dumps(current_metrics)}
Current parameters: {json.dumps(current_params)}
Last structure: {json.dumps(last_structure)}
Best structure: {json.dumps(best_structure)}

Suggest new hyperparameters for a GNN model for this dataset as a compact valid JSON dictionary, ready to use in PyTorch Geometric and Python (keys: layers, lr, weight_decay, optimizer, weight_initialization, dropout if needed).
Layers must be a list of dicts with type and out_channels.
Do NOT include Dropout as a layer in the layers list. If Dropout is used, set "dropout" as a top-level key.
Return JSON only.
""".strip()
    proposal = call_llm(prompt, llm_model)
    if 'error' not in proposal:
        STATE['best_params'] = proposal
    return proposal

@mcp.tool()
def suggest_new_params(
    llm_model: str,
    current_metrics: dict,
    current_params: dict,
    last_structure: dict,
    best_structure: dict,
    dataset_name: str
) -> dict:
    logger.info(f"[TOOL] suggest_new_params({llm_model}, ...dataset={dataset_name})")
    ds = get_dataset(dataset_name)
    ds_info = get_dataset_info(ds)
    prompt = f"""
You are an expert in GNN optimization. Here is the dataset info:
{json.dumps(ds_info, indent=2)}
Current metrics: {json.dumps(current_metrics)}
Current parameters: {json.dumps(current_params)}
Last structure: {json.dumps(last_structure)}
Best structure: {json.dumps(best_structure)}

Suggest improved model parameters as a valid JSON dictionary for PyTorch Geometric (keys: layers, lr, weight_decay, optimizer, weight_initialization, dropout if needed).
Do NOT include Dropout as a layer in the layers list. If Dropout is used, set "dropout" as a top-level key.
Return only JSON.
""".strip()
    proposal = call_llm(prompt, llm_model)
    if 'error' not in proposal:
        STATE['best_params'] = proposal
    return proposal

@mcp.tool()
def suggest_new_network(
    llm_model: str,
    current_metrics: dict,
    current_params: dict,
    last_structure: dict,
    best_structure: dict,
    dataset_name: str
) -> dict:
    logger.info(f"[TOOL] suggest_new_network({llm_model}, ...dataset={dataset_name})")
    ds = get_dataset(dataset_name)
    ds_info = get_dataset_info(ds)
    prompt = f"""
Given this graph dataset and model info:
{json.dumps(ds_info, indent=2)}
Current metrics: {json.dumps(current_metrics)}
Current parameters: {json.dumps(current_params)}
Last structure: {json.dumps(last_structure)}
Best structure: {json.dumps(best_structure)}

Suggest a new GNN architecture (layers as list of dicts with type, out_channels), and reasonable default hyperparams as JSON for PyTorch Geometric (including "dropout" as a top-level key if needed).
Do NOT include Dropout as a layer in the layers list. If Dropout is used, set "dropout" as a top-level key.
Return only JSON.
""".strip()
    proposal = call_llm(prompt, llm_model)
    if 'error' not in proposal:
        STATE['best_struct'] = proposal.get('structure', {})
        STATE['best_params'] = proposal
    return proposal

@mcp.tool()
def nas_search(dataset_name: str, llm_model: str, iterations: int = 10) -> dict:
    logger.info(f"[TOOL] nas_search({dataset_name}, {llm_model}, iterations={iterations})")
    best_metrics = None
    best_params = None
    for i in range(iterations):
        logger.info(f"[NAS] Iteration {i+1}/{iterations}")
        proposal = suggest_new_network(
            llm_model, {}, {}, {}, {}, dataset_name
        )
        result = run_iteration(dataset_name, proposal, 10)
        if (best_metrics is None) or (result['metrics']['test_accuracy'] > best_metrics['test_accuracy']):
            best_metrics = result['metrics']
            best_params = proposal
    STATE['best_metrics'] = best_metrics
    STATE['best_params'] = best_params
    return {"best_metrics": best_metrics, "best_params": best_params}

def main():
    configure_logger()
    parser = argparse.ArgumentParser(description="Start GNN Designer MCP Server")
    parser.add_argument('--server-type', choices=['sse', 'stdio'], default='sse')
    args = parser.parse_args()
    logger.info("🚀 Starting GNN Designer MCP Server...")
    mcp.run(args.server_type)

if __name__ == '__main__':
    main()

