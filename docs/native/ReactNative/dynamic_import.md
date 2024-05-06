# rn包的动态引入

众所周知，rn的核心特点就是动态引入，也就是可以自行codepush，从而绕过应用商店审核，那么动态引入到底是如何实现的那。简单来说如下：

-   下载bundle
-   加载bundle

## 简化版Android demo

```java title="MainActivity"
public class MainActivity extends AppCompatActivity {
    RecyclerView recyclerView;
    ListModule adapter;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        EdgeToEdge.enable(this);
        setContentView(R.layout.activity_main);
        initView();
        featchData();
    }

    /**
     * 初始化布局视图，默认数据为空
     */
    public void initView() {
        recyclerView = this.findViewById(R.id.list);
        recyclerView.setLayoutManager(new LinearLayoutManager(this));
        adapter = new ListModule(this, new ArrayList<ModuleItem.Bundle>());
        recyclerView.setAdapter(adapter);
        adapter.setOnItemClickListener(bundle -> {
            // 检查是否下载过，如果已经下载过则直接打开，暂不考虑各种版本问题
            String f = MainActivity.this.getFilesDir().getAbsolutePath() + "/" + bundle.name + "/" + bundle.name + ".bundle";
            File file = new File((f));
            if (file.exists()) {
                goToRNActivity(bundle.name);
            } else {
                download(bundle.name);
            }
        });
    }

    public void goToRNActivity(String bundleName) {
        Intent starter = new Intent(MainActivity.this, RNDynamicActivity.class);
        RNDynamicActivity.bundleName = bundleName;
        MainActivity.this.startActivity(starter);
    }

    /**
     * 调用服务获取数据
     */
    public void featchData() {
        OkHttpClient okHttpClient = new OkHttpClient();
        // 查询当前的bundle并在recycleview显示
        Request request = new Request.Builder().url(API.MODULES).method("GET", null).build();
        Call call = okHttpClient.newCall(request);
        call.enqueue(new Callback() {
            @Override
            public void onFailure(Call call, IOException e) {
                System.out.println("数据获取失败");
                System.out.println(e);
            }

            @Override
            public void onResponse(Call call, Response response) throws IOException {
                String data = response.body().string();

                ModuleItem moduleItem = new Gson().fromJson(data, ModuleItem.class);
                runOnUiThread(new Runnable() {

                    @Override
                    public void run() {
                        //刷新列表
                        adapter.clearModules();
                        // 添加bundle数据
                        adapter.addModules(moduleItem.data);
                    }
                });
            }
        });
    }

    /**
     * 点击跳转的时候进行下载
     * 下载对应的bundle
     *
     * @param bundleName
     */
    private void download(final String bundleName) {
        System.out.println(API.DOWNLOAD + bundleName);
        FileDownloader.setup(this);
        FileDownloader.getImpl().create(API.DOWNLOAD + bundleName).setPath(this.getFilesDir().getAbsolutePath(), true)

                .setListener(new FileDownloadListener() {
                    @Override
                    protected void started(BaseDownloadTask task) {
                        super.started(task);
                    }

                    @Override
                    protected void pending(BaseDownloadTask task, int soFarBytes, int totalBytes) {
                    }

                    @Override
                    protected void progress(BaseDownloadTask task, int soFarBytes, int totalBytes) {
                    }

                    @Override
                    protected void completed(BaseDownloadTask task) {

                        try {
                            //下载之后解压，然后打开
                            ZipUtils.unzip(MainActivity.this.getFilesDir().getAbsolutePath() + "/" + bundleName + ".zip", MainActivity.this.getFilesDir().getAbsolutePath());

                            goToRNActivity(bundleName);

                        } catch (Exception e) {
                            e.printStackTrace();
                        }

                    }

                    @Override
                    protected void paused(BaseDownloadTask task, int soFarBytes, int totalBytes) {
                    }

                    @Override
                    protected void error(BaseDownloadTask task, Throwable e) {
                    }

                    @Override
                    protected void warn(BaseDownloadTask task) {

                    }
                }).start();
    }
}
```

因为是最简单的动态化测试demo，所以并没有引入下载时机，patch对比，最小下载等概念。后续继续完善，但是通过该bundle也能看到rn动态化的基本实现~