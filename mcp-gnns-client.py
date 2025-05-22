import asyncio
import json
import sys
import shlex
from llama_index.tools.mcp import BasicMCPClient, McpToolSpec

HELP_TEXT = """
Przykłady użycia (kopiuj-wklej do terminala!):

1. Pobierz statystyki datasetu:
   dataset_info Cora

2. Przeprowadź jedną iterację trenowania:
   run_iteration Cora '{"layers":[{"type":"GCNConv","out_channels":16}],"lr":0.01,"weight_decay":5e-4,"optimizer":"Adam","weight_initialization":"xavier"}' 10

3. Zasugeruj nowe hiperparametry (LLM/Ollama):
   suggest_hyperparams llama3.2 '{"test_accuracy":0.82}' '{"layers":[{"type":"GCNConv","out_channels":16}]}' '{"layers":["GCNConv"]}' '{"layers":["GCNConv"]}' Cora

4. Zasugeruj nową architekturę sieci (LLM/Ollama):
   suggest_new_network llama3.2 '{"test_accuracy":0.82}' '{"layers":[{"type":"GCNConv","out_channels":16}]}' '{"layers":["GCNConv"]}' '{"layers":["GCNConv"]}' Cora

5. Zasugeruj nowe parametry (LLM/Ollama):
   suggest_new_params llama3.2 '{"test_accuracy":0.82}' '{"layers":[{"type":"GCNConv","out_channels":16}]}' '{"layers":["GCNConv"]}' '{"layers":["GCNConv"]}' Cora

6. Pobierz ostatnie użyte hiperparametry:
   get_params

7. Pobierz najlepsze hiperparametry (LLM):
   get_hyperparams

8. Pobierz najlepszą architekturę (LLM):
   get_best_architecture

9. Pobierz najlepsze metryki (LLM):
   get_best_results

10. Pobierz najlepsze hiperparametry (LLM):
    get_best_hyperparams

11. Przeprowadź trening na najlepszych hiperparametrach:
    train_best Cora 10

12. Przeszukaj NAS (Neural Architecture Search):
    nas_search Cora llama3.2 10

13. Pomoc:
    help

14. Wyjście:
    exit

Legenda argumentów:
- dataset_name — np. Cora, Citeseer, PubMed, itd.
- hyperparams — słownik (dict) z parametrami modelu, np. '{"layers":[{"type":"GCNConv","out_channels":16}],...}'
- epochs — liczba epok, np. 10
- llm_model — np. llama3:2
"""

NO_LLM_TOOLS = {
    'dataset_info', 'run_iteration', 'get_params', 'get_best_architecture',
    'get_best_results', 'get_best_hyperparams', 'get_hyperparams', 'train_best'
}
LLM_TOOLS = {
    'suggest_hyperparams', 'suggest_new_params', 'suggest_new_network', 'nas_search'
}

# Precyzyjne mapowanie argumentów do narzędzi (dla *acall)
TOOL_ARGNAMES = {
    'dataset_info': ['dataset_name'],
    'run_iteration': ['dataset_name', 'hyperparams', 'epochs'],
    'get_params': [],
    'get_best_architecture': [],
    'get_hyperparams': [],
    'get_best_results': [],
    'get_best_hyperparams': [],
    'train_best': ['dataset_name', 'epochs'],
    'suggest_hyperparams': [
        'llm_model', 'current_metrics', 'current_params', 'last_structure', 'best_structure', 'dataset_name'
    ],
    'suggest_new_params': [
        'llm_model', 'current_metrics', 'current_params', 'last_structure', 'best_structure', 'dataset_name'
    ],
    'suggest_new_network': [
        'llm_model', 'current_metrics', 'current_params', 'last_structure', 'best_structure', 'dataset_name'
    ],
    'nas_search': ['dataset_name', 'llm_model', 'epochs']
}

def tool_result_to_json(result):
    # Konwertuj ToolOutput lub inne klasy do JSON-serializowalnego obiektu
    if hasattr(result, 'content'):
        # Najczęściej to TextContent lub podobny, zwróć jako tekst lub parsuj jako JSON
        txt = result.content if isinstance(result.content, str) else getattr(result.content[0], "text", str(result.content))
        try:
            return json.loads(txt)
        except Exception:
            return txt
    if isinstance(result, (dict, list)):
        return result
    # Spróbuj zdekodować string
    try:
        return json.loads(str(result))
    except Exception:
        return str(result)

async def main():
    mcp_client = BasicMCPClient("http://127.0.0.1:8000/sse")
    tools = await McpToolSpec(client=mcp_client).to_tool_list_async()
    tools_dict = {getattr(tool, 'name', tool.metadata.name): tool for tool in tools}
    print("🤖 GNN Designer ready. Type 'help' for examples or 'exit' to quit.\n")
    print(HELP_TEXT)

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not user_input:
            continue

        cmd = user_input.split()[0]
        if cmd == 'exit':
            break
        if cmd == 'help':
            print(HELP_TEXT)
            continue

        tool = tools_dict.get(cmd)
        if not tool:
            print(json.dumps({"error": f"Unknown command {cmd}"}))
            continue

        try:
            args = []
            if len(user_input.split(maxsplit=1)) > 1:
                parts = shlex.split(user_input)
                args = parts[1:]
                for i, a in enumerate(args):
                    try:
                        if (a.startswith('{') or a.startswith('[')):
                            args[i] = json.loads(a)
                    except Exception:
                        pass

            argnames = TOOL_ARGNAMES.get(cmd, [])
            call_kwargs = {k: v for k, v in zip(argnames, args)}

            # Narzędzia bez LLM
            if cmd in NO_LLM_TOOLS:
                result = await tool.acall(**call_kwargs)
                out = tool_result_to_json(result)
                print(json.dumps(out, indent=2, ensure_ascii=False) if isinstance(out, (dict, list)) else out)
            # Narzędzia przez LLM: najpierw LLM, potem iteration!
            elif cmd in LLM_TOOLS:
                suggest_result = await tool.acall(**call_kwargs)
                suggest_out = tool_result_to_json(suggest_result)
                print("[INFO] LLM suggested params: ", json.dumps(suggest_out, indent=2, ensure_ascii=False) if isinstance(suggest_out, (dict, list)) else suggest_out)
                # Jeśli jest to narzędzie typu nas_search, nie robimy run_iteration
                if cmd == "nas_search":
                    continue
                # Wyciągnij hiperparametry do iteration, jeśli dotyczy
                if isinstance(suggest_out, dict) and 'layers' in suggest_out:
                    dataset_name = args[-1]
                    epochs = 10
                    run_iteration_tool = tools_dict.get('run_iteration')
                    if not run_iteration_tool:
                        print(json.dumps({"error": "run_iteration tool not available!"}))
                        continue
                    iter_kwargs = {
                        'dataset_name': dataset_name,
                        'hyperparams': suggest_out,
                        'epochs': epochs
                    }
                    result = await run_iteration_tool.acall(**iter_kwargs)
                    out = tool_result_to_json(result)
                    print(json.dumps(out, indent=2, ensure_ascii=False) if isinstance(out, (dict, list)) else out)
            else:
                print(json.dumps({"error": "Unsupported command"}))
        except Exception as e:
            print(json.dumps({"error": str(e)}))

    sys.exit(0)

if __name__ == '__main__':
    asyncio.run(main())

