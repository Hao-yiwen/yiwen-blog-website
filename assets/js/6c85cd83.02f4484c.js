"use strict";(self.webpackChunkyiwen_blog_website=self.webpackChunkyiwen_blog_website||[]).push([[5522],{3905:(e,t,n)=>{n.d(t,{Zo:()=>s,kt:()=>g});var a=n(67294);function r(e,t,n){return t in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}function i(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var a=Object.getOwnPropertySymbols(e);t&&(a=a.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),n.push.apply(n,a)}return n}function o(e){for(var t=1;t<arguments.length;t++){var n=null!=arguments[t]?arguments[t]:{};t%2?i(Object(n),!0).forEach((function(t){r(e,t,n[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):i(Object(n)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))}))}return e}function l(e,t){if(null==e)return{};var n,a,r=function(e,t){if(null==e)return{};var n,a,r={},i=Object.keys(e);for(a=0;a<i.length;a++)n=i[a],t.indexOf(n)>=0||(r[n]=e[n]);return r}(e,t);if(Object.getOwnPropertySymbols){var i=Object.getOwnPropertySymbols(e);for(a=0;a<i.length;a++)n=i[a],t.indexOf(n)>=0||Object.prototype.propertyIsEnumerable.call(e,n)&&(r[n]=e[n])}return r}var c=a.createContext({}),d=function(e){var t=a.useContext(c),n=t;return e&&(n="function"==typeof e?e(t):o(o({},t),e)),n},s=function(e){var t=d(e.components);return a.createElement(c.Provider,{value:t},e.children)},m="mdxType",p={inlineCode:"code",wrapper:function(e){var t=e.children;return a.createElement(a.Fragment,{},t)}},u=a.forwardRef((function(e,t){var n=e.components,r=e.mdxType,i=e.originalType,c=e.parentName,s=l(e,["components","mdxType","originalType","parentName"]),m=d(n),u=r,g=m["".concat(c,".").concat(u)]||m[u]||p[u]||i;return n?a.createElement(g,o(o({ref:t},s),{},{components:n})):a.createElement(g,o({ref:t},s))}));function g(e,t){var n=arguments,r=t&&t.mdxType;if("string"==typeof e||r){var i=n.length,o=new Array(i);o[0]=u;var l={};for(var c in t)hasOwnProperty.call(t,c)&&(l[c]=t[c]);l.originalType=e,l[m]="string"==typeof e?e:r,o[1]=l;for(var d=2;d<i;d++)o[d]=n[d];return a.createElement.apply(null,o)}return a.createElement.apply(null,n)}u.displayName="MDXCreateElement"},37119:(e,t,n)=>{n.r(t),n.d(t,{assets:()=>c,contentTitle:()=>o,default:()=>p,frontMatter:()=>i,metadata:()=>l,toc:()=>d});var a=n(87462),r=(n(67294),n(3905));const i={},o="Activity\u6dfb\u52a0Fragment",l={unversionedId:"native/android/javaView/activity_add_fragment",id:"native/android/javaView/activity_add_fragment",title:"Activity\u6dfb\u52a0Fragment",description:"\u4eca\u5929\u5728JavaView\u4e2d\u6dfb\u52a0\u6dfb\u52a0ToolsBar,\u7136\u540e\u53d1\u73b0\u5982\u679c\u4e0d\u8fdb\u884c\u5c01\u88c5\uff0c\u6bcf\u4e00\u4e2a\u9875\u9762\u5ea6\u9700\u8981\u6dfb\u52a0ToolsBar,\u6240\u4ee5\u5199\u4e86\u4e00\u4e2aFragment\u6765\u5c01\u88c5ToolsBar,\u4e5f\u8bb0\u5f55\u4e00\u4e0bActivity\u6dfb\u52a0Fragment.",source:"@site/docs/native/android/javaView/activity_add_fragment.md",sourceDirName:"native/android/javaView",slug:"/native/android/javaView/activity_add_fragment",permalink:"/yiwen-blog-website/docs/native/android/javaView/activity_add_fragment",draft:!1,editUrl:"https://github.com/Hao-yiwen/yiwen-blog-website/tree/master/docs/native/android/javaView/activity_add_fragment.md",tags:[],version:"current",frontMatter:{},sidebar:"nativeSidebar",previous:{title:"Java View",permalink:"/yiwen-blog-website/docs/category/java-view"},next:{title:"ConstraintLayout - \u7ea6\u675f\u5e03\u5c40",permalink:"/yiwen-blog-website/docs/native/android/javaView/constraint_layout"}},c={},d=[{value:"\u58f0\u660eFragment.xml",id:"\u58f0\u660efragmentxml",level:2},{value:"\u7f16\u5199Fragment.java",id:"\u7f16\u5199fragmentjava",level:2},{value:"\u5728activity\u4e2d\u4f7f\u7528",id:"\u5728activity\u4e2d\u4f7f\u7528",level:2},{value:"\u5728Activity.java\u4e2d\u6dfb\u52a0",id:"\u5728activityjava\u4e2d\u6dfb\u52a0",level:2}],s={toc:d},m="wrapper";function p(e){let{components:t,...n}=e;return(0,r.kt)(m,(0,a.Z)({},s,n,{components:t,mdxType:"MDXLayout"}),(0,r.kt)("h1",{id:"activity\u6dfb\u52a0fragment"},"Activity\u6dfb\u52a0Fragment"),(0,r.kt)("p",null,"\u4eca\u5929\u5728JavaView\u4e2d\u6dfb\u52a0\u6dfb\u52a0ToolsBar,\u7136\u540e\u53d1\u73b0\u5982\u679c\u4e0d\u8fdb\u884c\u5c01\u88c5\uff0c\u6bcf\u4e00\u4e2a\u9875\u9762\u5ea6\u9700\u8981\u6dfb\u52a0ToolsBar,\u6240\u4ee5\u5199\u4e86\u4e00\u4e2aFragment\u6765\u5c01\u88c5ToolsBar,\u4e5f\u8bb0\u5f55\u4e00\u4e0bActivity\u6dfb\u52a0Fragment."),(0,r.kt)("h2",{id:"\u58f0\u660efragmentxml"},"\u58f0\u660eFragment.xml"),(0,r.kt)("pre",null,(0,r.kt)("code",{parentName:"pre",className:"language-xml"},'<?xml version="1.0" encoding="utf-8"?>\n<FrameLayout xmlns:android="http://schemas.android.com/apk/res/android"\n    xmlns:tools="http://schemas.android.com/tools"\n    android:layout_width="match_parent"\n    android:layout_height="wrap_content"\n    tools:context=".fragment.ToolBarFragment">\n\n    <androidx.appcompat.widget.Toolbar\n        android:id="@+id/toolbar_fragment"\n        android:layout_width="match_parent"\n        android:layout_height="match_parent"\n        android:background="?attr/colorPrimary"\n        android:elevation="4dp"\n        android:theme="@style/ThemeOverlay.AppCompat.Dark.ActionBar" />\n\n</FrameLayout>\n')),(0,r.kt)("h2",{id:"\u7f16\u5199fragmentjava"},"\u7f16\u5199Fragment.java"),(0,r.kt)("pre",null,(0,r.kt)("code",{parentName:"pre",className:"language-java"},'public class ToolBarFragment extends Fragment {\n\n    public ToolBarFragment() {\n        // Required empty public constructor\n    }\n\n    public static ToolBarFragment newInstance(String param1) {\n        ToolBarFragment fragment = new ToolBarFragment();\n        Bundle args = new Bundle();\n        args.putString("TITLE", param1);\n        fragment.setArguments(args);\n        return fragment;\n    }\n\n    @Override\n    public void onCreate(Bundle savedInstanceState) {\n        super.onCreate(savedInstanceState);\n    }\n\n    @Override\n    public View onCreateView(LayoutInflater inflater, ViewGroup container,\n                             Bundle savedInstanceState) {\n        View view = inflater.inflate(R.layout.fragment_tool_bar, container, false);\n        Toolbar toolbar = view.findViewById(R.id.toolbar_fragment);\n\n        if (getArguments() != null && getArguments().containsKey("TITLE")) {\n            String title = getArguments().getString("TITLE");\n            toolbar.setTitle(title);\n        }\n\n        return view;\n    }\n}\n')),(0,r.kt)("h2",{id:"\u5728activity\u4e2d\u4f7f\u7528"},"\u5728activity\u4e2d\u4f7f\u7528"),(0,r.kt)("pre",null,(0,r.kt)("code",{parentName:"pre",className:"language-xml"},'<FrameLayout\n    android:id="@+id/fragment_toolbar"\n    android:layout_width="match_parent"\n    android:layout_height="wrap_content"\n    app:layout_constraintTop_toTopOf="parent" />\n')),(0,r.kt)("h2",{id:"\u5728activityjava\u4e2d\u6dfb\u52a0"},"\u5728Activity.java\u4e2d\u6dfb\u52a0"),(0,r.kt)("pre",null,(0,r.kt)("code",{parentName:"pre",className:"language-xml"},'@Override\nprotected void onCreate(Bundle savedInstanceState) {\n    super.onCreate(savedInstanceState);\n    EdgeToEdge.enable(this);\n    setContentView(R.layout.activity_list_view);\n\n    /**\n        @description \u6dfb\u52a0fragment\n    */\n    if (savedInstanceState == null) {\n        getSupportFragmentManager().beginTransaction().add(R.id.fragment_toolbar, ToolBarFragment.newInstance("ListView")).commit();\n    }\n}\n')))}p.isMDXComponent=!0}}]);