import os
import yaml

config_dir = 'config'
for fname in os.listdir(config_dir):
    if fname.endswith('.yml'):
        path = os.path.join(config_dir, fname)
        with open(path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        cfg['training-ratio'] = 10
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(cfg, f, allow_unicode=True)
print('所有config已批量设置 training-ratio: 10')