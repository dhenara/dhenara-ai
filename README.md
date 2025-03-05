```markdown
# Dhenara

Python package for interacting with various AI models in a unified way.

[![PyPI version](https://badge.fury.io/py/dhenara.svg)](https://badge.fury.io/py/dhenara)
[![Build Status](https://github.com/dhenara/dhenara/actions/workflows/tests.yml/badge.svg)](https://github.com/dhenara/dhenara/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Installation

```bash
pip install dhenara
```

For AWS support:
```bash
pip install "dhenara[aws]"
```

For Azure support:
```bash
pip install "dhenara[azure]"
```

## Quick Start

```python
from dhenara.ai import ChatModel

model = ChatModel.from_provider("openai", model="gpt-4")
response = model.generate("Hello, world!")
print(response)
```

## Documentation

For full documentation, visit [docs.dhenara.com](https://docs.dhenara.com/).