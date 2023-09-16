---
sidebar_position: 6
---

# 断点续传

断点续传是一种网络传输技术，用于实现文件的部分下载或上传，以便在连接断开或其他故障发生时，能从中断的位置继续进行，而不是重新开始。这在传输大文件或在网络环境不稳定的情况下特别有用。

## 常用的断点续传技术：

HTTP Range 请求：HTTP/1.1 协议支持 Range 头，用于请求文件的某个范围。服务端也会返回相应的范围数据，并使用 206 Partial Content 状态码作为响应。

分块下载：将文件分成多个小块，每个小块都可以独立下载。如果某个块下载失败，只需重新下载该块。

P2P 传输：在 Peer-to-Peer 网络中，同一个文件可能存在于多个节点上。断点续传可以通过从不同节点下载不同的文件块来实现。

FTP 协议：FTP（文件传输协议）也支持断点续传，通常通过 REST 命令来设置开始传输的文件偏移量。

元数据记录：通过记录已经成功下载或上传的数据块的信息，可以在程序或系统重启后，准确地从上次成功的地方继续。

## 项目选型

分块上传下载。

```java
//    分块下载
    @Test
    public void testChunk() throws Exception {
        File sourceFile = new File("/Users/yw.hao/Desktop/2.mov");
//        分块文件存储路径
        String chunkFilePath = "/Users/yw.hao/Desktop/chunk/";
//        分开文件大小
        int chunkSize = 1024 * 1024 * 1;
//        分开文件个数
        int chunkNum = (int) Math.ceil(sourceFile.length() * 1.0 / chunkSize);
        RandomAccessFile r = new RandomAccessFile(sourceFile, "r");
//        缓冲区
        byte[] bytes = new byte[1024];
        for (int i = 0; i < chunkNum; i++) {
            File chunkFile = new File(chunkFilePath + i);
            RandomAccessFile raf_r = new RandomAccessFile(chunkFile, "rw");
            int len = -1;
            while ((len = r.read(bytes)) != -1) {
                raf_r.write(bytes, 0, len);
                if (chunkFile.length() >= chunkSize) {
                    break;
                }
            }
            raf_r.close();
        }
        r.close();
    }

    //    分块合并
    @Test
    public void testMerge() throws IOException {
        //        分块文件存储路径
        String chunkFilePath = "/Users/yw.hao/Desktop/chunk/";
        File chunkFolder = new File(chunkFilePath);
        File sourceFile = new File("/Users/yw.hao/Desktop/2.mov");
        File mergeFile = new File("/Users/yw.hao/Desktop/3.mov");
        File[] files = chunkFolder.listFiles();
        List<File> fileList = Arrays.asList(files);

        Collections.sort(fileList, new Comparator<File>() {
            @Override
            public int compare(File o1, File o2) {
                return Integer.parseInt(o1.getName()) - Integer.parseInt(o2.getName());
            }
        });
//        向合并文件写的流
        RandomAccessFile raf_rw = new RandomAccessFile(mergeFile, "rw");
        byte[] bytes = new byte[1024];

//        便利分块文件，合并
        for (File file : fileList) {
            RandomAccessFile raf_r = new RandomAccessFile(file, "r");
            int len = -1;
            while ((len = raf_r.read(bytes)) != -1) {
                raf_rw.write(bytes, 0, len);
            }
            raf_r.close();
        }
        raf_rw.close();
//        合并文件完成
        FileInputStream fileInputStream_merge = new FileInputStream(mergeFile);
        FileInputStream fileInputStream_source = new FileInputStream(sourceFile);
        String md5_merge = DigestUtils.md5Hex(fileInputStream_merge);
        String md5_source = DigestUtils.md5Hex(fileInputStream_source);
        if (md5_merge.equals(md5_source)) {
            System.out.println("文件合并完成");
        }
    }
```
