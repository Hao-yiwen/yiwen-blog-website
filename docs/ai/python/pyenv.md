# pyenv报错

```bash
844 Segmentation fault: 11 "$PYENV_COMMAND_PATH" "$@
```

在安装`torch`的时候遇到上述问题，各种方案死活不生效。

经过各类尝试发现是macos升级会导致pyenv失效，此时需要重新安装`brew`。

issue: https://github.com/pyenv/pyenv/issues/1048
