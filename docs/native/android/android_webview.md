---
title: Android添加webview
sidebar_label: Android添加webview
date: 2024-06-25
last_update:
  date: 2024-06-25
---

# Android添加webview

1. 创建activity，activity包含webview
```java title="MainActivity.java"
import android.os.Bundle;
import android.webkit.WebChromeClient;
import android.webkit.WebView;
import android.webkit.WebViewClient;
import androidx.appcompat.app.AppCompatActivity;

public class MainActivity extends AppCompatActivity {
    private WebView webView;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        webView = findViewById(R.id.webview);
        webView.getSettings().setJavaScriptEnabled(true);
        webView.setWebViewClient(new WebViewClient());
        webView.setWebChromeClient(new WebChromeClient());

        // Load a local HTML file or a remote URL
        webView.loadUrl("file:///android_asset/index.html");

        // Add the JavaScript interface
        webView.addJavascriptInterface(new WebAppInterface(this), "Android");
    }
}
```

2. 实现js接口
```java
// WebAppInterface.java
import android.content.Context;
import android.webkit.JavascriptInterface;
import android.widget.Toast;

public class WebAppInterface {
    Context mContext;

    /** Instantiate the interface and set the context */
    WebAppInterface(Context c) {
        mContext = c;
    }

    /** Show a toast from the web page */
    @JavascriptInterface
    public void showToast(String toast) {
        Toast.makeText(mContext, toast, Toast.LENGTH_SHORT).show();
    }

    /** Add more methods here that you want to expose to JavaScript */
}
```
3. html调用js接口
```html
<!DOCTYPE html>
<html>
<head>
    <title>WebView Bridge</title>
    <script type="text/javascript">
        window.onload = function() {
            function showToast() {
                console.log("Button clicked");
                if(window.Android){
                    Android.showToast("Hello from JavaScript!");
                } else {
                    alert("Android object not found. Make sure you are running in WebView.");
                }
            }
            document.getElementById("button").addEventListener("click", showToast);
        }
    </script>
</head>
<body>
<h1>WebView Bridge Example</h1>
<button type="button" id="button">Show Toast</button>
</body>
</html>
```