# LLMetal

!!!LouLouMetal!!!

## Builds
Builds are done using cmake and ninja. First, ensure you have the following installed, otherwise install them (e.g. for MacOS, use `xcode-select --install && brew install cmake ninja`):
```bash
clang++ --version 
cmake --version 
ninja --version
```

You can then build and run manually from the root directory using cmake cli commands:

```bash
# Other presets are profile and release
cmake --preset debug
cmake --build --preset debug
```

Then, you can run it using:
```bash
./build/debug/LLMetal
```
