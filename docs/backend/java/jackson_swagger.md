# jackson导致swagger无法正常工作

```java
@Slf4j
@Configuration
public class WebMvcConfig extends WebMvcConfigurationSupport {

    @Override
    protected void extendMessageConverters(List<HttpMessageConverter<?>> converters) {
        log.info("扩展消息转换器");
        MappingJackson2HttpMessageConverter messageConverter = new MappingJackson2HttpMessageConverter();
        // 设置对象转换器，底层使用jackson将java对象转为json
        messageConverter.setObjectMapper(new JacksonObjectMapper());
        converters.add(0, messageConverter);
    }
}
```

`spring boot3.2`中`springdoc-openapi-starter-webmvc-ui`库无法正常使用，经过漫长的debug后发现和`Jackson`有关，也就是swagger底层的jackson和自定义的对象转换器有冲突，从而导致swagger无法正常使用。
```gradle
implementation 'org.springdoc:springdoc-openapi-starter-webmvc-ui:2.3.0'
```