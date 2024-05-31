# Android集成rn常见报错

## command node path问题

```bash
Error:Execution failed for task ':app:recordFilesBeforeBundleCommandDebug'.
> A problem occurred starting process 'command 'node''
```

-   解决:

```
./gradlew --stop
```
