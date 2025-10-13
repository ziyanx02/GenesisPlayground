# Real-Robot Interface

This folder contains the real-robot runtime handlers (e.g., low-state readers, controllers).
Previously we vendored a `cyclonedds` submodule. To simplify the repo, the submodule was
removed. If you need real-robot features, install the external dependencies as below.

> TL;DR: Install Unitree SDK2 (Python) and CycloneDDS from source, then run the examples.

---

## 1) Install Unitree SDK2 (Python)

```bash
cd ~
sudo apt update
sudo apt install -y python3-pip
git clone https://github.com/unitreerobotics/unitree_sdk2_python.git
cd unitree_sdk2_python
pip3 install -e .
