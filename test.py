from pathlib import Path
import json

p = Path('data/processed/tardis/phase1_outputs/phase1_prompts.json')
records = json.loads(p.read_text())

# Lấy một record của contrarian phase=drop
for r in records:
    if r['agent_type'] == 'contrarian_trader' and r['phase'] == 'drop':
        print("=== SYSTEM PROMPT (first 800 chars) ===")
        print(r['system_prompt'][:800])
        print("\n=== USER PROMPT (targets section) ===")
        idx = r['user_prompt'].find('Agent-specific elicitation')
        print(r['user_prompt'][idx:idx+600])
        break