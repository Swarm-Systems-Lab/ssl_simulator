# Swarm Systems Lab Python Simulator

## Installation

You can install the ```ssl_simulator``` package directly from GitHub.

> The ```ssl_simulator``` package provides a stable, tested set of dependencies for the following packages: ```numpy```, ```matplotlib```, ```tqdm```, ```pandas```, ```scipy```, ```ipython```. We do not recommend modifying the version ranges, as the package has been tested to work with this specific combination.

Option 1. Install the latest **development version**
```bash
pip install git+https://github.com/Swarm-Systems-Lab/ssl_simulator.git@master
```
Or, add it to your ```requierements.txt```:
```txt
git+https://github.com/Swarm-Systems-Lab/ssl_simulator.git@master
```
Option 2. Install a specific **stable release**

To use a stable version instead of the latest development code, replace ```@master``` with a release tag, e.g.:
```bash
pip install git+https://github.com/Swarm-Systems-Lab/ssl_simulator.git@v0.0.1
```
Or in ```requierements.txt```:
```txt
git+https://github.com/Swarm-Systems-Lab/ssl_simulator.git@v0.0.1
```

Option 3. Editable **local install** for development
   
If you cloned the repository locally and want changes to the code to be immediately reflected without reinstalling:
```bash
git clone https://github.com/Swarm-Systems-Lab/ssl_simulator.git
cd ssl_simulator
pip install -e .
```

### Additional Dependencies

1. **FFmpeg** (Required for animations)

    * Linux (Debian/Ubuntu-based distributions):
    ```
    sudo apt-get update && sudo apt-get -y install ffmpeg
    ```
    * MacOS (via Homebrew):
    ```
    brew install ffmpeg
    ```
    * Windows: Download and install FFmpeg from https://ffmpeg.org/download.html and ensure it is added to your system's PATH.


2. **LaTeX Fonts** (Required for some visualization tools)

    * Linux (Debian/Ubuntu-based distributions):
    ```
    sudo apt-get update && sudo apt-get install -y texlive texlive-latex-extra texlive-fonts-recommended dvipng cm-super
    ```
    
    * MacOS (via MacTeX):
    ```
    brew install mactex
    ```

    * Windows: Install [MiKTeX](https://miktex.org/) or TeX Live.

## 🚧 Documentation in Progress 🚧

⚠️ **Warning:** The documentation section of this repository is currently a **work in progress**. Some details may be missing, incomplete, or subject to change. We are actively working to improve and expand the README to provide clear instructions, examples, and usage guidelines.

In the meantime, feel free to explore the code and contribute with suggestions. If you have any questions, open an issue or reach out to the maintainers:

- **[Jesús Bautista Villar](https://sites.google.com/view/jbautista-research)** (<jesbauti20@gmail.com>) – Main Developer


Thanks for your patience! 🚀
