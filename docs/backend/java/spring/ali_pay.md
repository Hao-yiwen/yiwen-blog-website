---
sidebar_position: 9
---

# 支付宝支付

## 文档

[支付宝支付文档](https://open.alipay.com/develop/sandbox/account)

## 说明

支付宝提供了沙箱支付宝供开发者开发,在测试环境开发者可通过沙箱app进行开发调试，为了进一步降低开发者开发成本，开发者在测试时可以模拟手机网页测试，同样可模拟整个测试流程。
[沙箱支付宝](https://opendocs.alipay.com/common/07enlc)

## 依赖说明

```xml
<!-- 支付宝SDK -->
<dependency>
    <groupId>com.alipay.sdk</groupId>
    <artifactId>alipay-sdk-java</artifactId>
    <version>3.7.73.ALL</version>
</dependency>

<!-- 支付宝SDK依赖的日志 -->
<dependency>
    <groupId>commons-logging</groupId>
    <artifactId>commons-logging</artifactId>
    <version>1.2</version>
</dependency>
```

## 示例代码

```java
@GetMapping("/requestpay")
public void requestpay(String payNo, HttpServletResponse httpResponse) throws IOException, AlipayApiException {
    // 请求支付宝去下单
    AlipayClient alipayClient = new DefaultAlipayClient(AlipayConfig.URL, APP_ID, APP_PRIVATE_KEY, AlipayConfig.FORMAT, AlipayConfig.CHARSET, ALIPAY_PUBLIC_KEY, AlipayConfig.SIGNTYPE);
    AlipayTradeWapPayRequest alipayRequest = new AlipayTradeWapPayRequest();//创建API对应的request
    //        alipayRequest.setReturnUrl("http://domain.com/CallBack/return_url.jsp");
    alipayRequest.setNotifyUrl(" https://e099-101-88-68-235.ngrok.io/orders/paynotify");//在公共参数中设置回跳和通知地址
    alipayRequest.setBizContent("{" +
            "    \"out_trade_no\":\"" + payNo + "\"," +
            "    \"total_amount\":" + payRecordByPayno.getTotalPrice() + "," +
            "    \"subject\":\"" + payRecordByPayno.getOrderName() + "\"," +
            "    \"product_code\":\"QUICK_MSECURITY_PAY\"" +
            "  }");//填充业务参数
    String form = alipayClient.pageExecute(alipayRequest).getBody(); //调用SDK生成表单
    httpResponse.setContentType("text/html;charset=" + AlipayConfig.CHARSET);
    httpResponse.getWriter().write(form);//直接将完整的表单html输出到页面
    httpResponse.getWriter().flush();
}
```

## 内网穿透工具

在扫码登录或者支付场景，需要第三方服务回调本地服务，所以此时需要内网穿透来供外部服务调用。

```bash
brew install ngrok/ngrok/ngrok
```
