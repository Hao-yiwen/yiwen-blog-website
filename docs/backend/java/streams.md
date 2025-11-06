---
title: Java中的各种流
sidebar_label: Java中的各种流
date: 2024-06-25
last_update:
  date: 2024-06-25
---

# Java中的各种流

-   字节流（Byte Streams）: 字节流主要用于处理原始二进制数据。它们是所有输入输出流的基础，可以用来读取或写入任何类型的数据，不仅仅是文本。字节流在java.io包中，最基本的类是InputStream和OutputStream。这些流对于处理图像、音频、视频等非文本数据非常有用。

-   网络流（Network Streams）: 网络流是用于处理网络数据传输的。在Java中，通过套接字（Socket）进行网络通信时，通常会用到网络流。例如，Socket类提供了getInputStream()和getOutputStream()方法来读取和写入数据，这些就是网络流的实例。网络流允许你通过网络发送或接收数据，例如，在客户端和服务器之间传输文件或消息。

-   文件流（File Streams）: 文件流专门用于读取和写入文件数据。在java.io包中，FileInputStream和FileOutputStream是用于处理文件数据的基本类。文件流使得可以直接从文件系统中读取或写入文件，无论文件内容是文本还是二进制数据。

-   字符流（Character Streams）: 字符流主要用于处理文本数据。它们是字节流的扩展，提供了字符级别的操作。在java.io包中，Reader和Writer是所有字符流的超类。字符流自动处理字符编码和解码，使得读写文本文件更加容易。常用的字符流包括FileReader、FileWriter、InputStreamReader和OutputStreamWriter。

-   缓冲流（Buffered Streams）: 缓冲流通过设置内部缓冲区，提高了读写数据的效率。在java.io包中，BufferedReader和BufferedWriter用于文本数据，而BufferedInputStream和BufferedOutputStream用于字节数据。缓冲流可以减少实际进行的物理读写操作，从而提高性能。

-   数据流（Data Streams）: 数据流用于处理原始数据类型（如int, float, double等）以及String对象的便捷读写。DataInputStream和DataOutputStream类提供了读取和写入基本数据类型的方法，如readInt(), writeDouble()等。这些流常用于读写格式化的数据。

-   对象流（Object Streams）: 对象流允许读取或写入对象。ObjectInputStream和ObjectOutputStream类可以对对象进行序列化和反序列化，即将对象转换为字节序列和从字节序列重构对象。这对于对象持久化和远程方法调用非常重要。

-   管道流（Piped Streams）: 管道流用于在不同线程之间传输数据。PipedInputStream和PipedOutputStream（用于字节数据），以及PipedReader和PipedWriter（用于字符数据），可以使一个线程的输出成为另一个线程的输入。

-   打印流（Print Streams）: PrintStream和PrintWriter类提供了格式化打印对象和文本数据的便捷方法。这些流可以用于生成格式化的输出，类似于C语言中的printf功能。

-   序列输入流（SequenceInputStream）: SequenceInputStream允许将多个输入流串联成一个输入流。这对于从多个源连续读取数据非常有用。

## 缓冲流

在现代Java编程中，确实更常见的做法是使用缓冲流（如BufferedReader和BufferedWriter）来读写文件，而不是直接使用基础的字节流（如FileInputStream和FileOutputStream）或字符流（如FileReader和FileWriter）。这是因为缓冲流提供了几个重要的优势：

-   性能提升：缓冲流通过内部缓冲机制减少了实际的物理读写操作次数。这意味着数据会先被累积到缓冲区中，然后在缓冲区满时一次性进行物理写操作，或者从缓冲区中一次性读取多个数据，这大大减少了磁盘或网络IO操作，从而提高性能。

-   方便的方法：缓冲流提供了一些额外的便利方法，例如BufferedReader的readLine()方法可以一次读取一行数据，而BufferedWriter的newLine()方法可以方便地写入行分隔符。

-   减少资源占用：频繁的磁盘或网络IO操作会增加资源的占用。通过减少这些操作的频率，缓冲流有助于更有效地管理资源。

不过，这并不意味着字节流和字符流没有它们的用处。在一些特定场景中，直接使用字节流和字符流可能更为合适：

-   当处理的数据量非常小，或者需要对数据的读写进行精细控制时，直接使用基础流可能更简单直接。
-   在某些性能敏感的应用中，直接管理字节流可以更精确地控制数据处理的方式，从而实现最优性能。
-   当需要处理原始二进制数据时，使用字节流（如FileInputStream和FileOutputStream）是必要的。

总之，是否使用缓冲流取决于具体的应用场景和性能要求。在大多数情况下，使用缓冲流由于其性能优势和使用便利性，是推荐的做法。

## 缓冲流示例

```java
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;

public class BufferedReadWriteExample {
    public static void main(String[] args) {
        String inputFile = "input.txt"; // 输入文件名
        String outputFile = "output.txt"; // 输出文件名

        try (
            // 创建BufferedReader和BufferedWriter，使用try-with-resources自动关闭资源
            BufferedReader reader = new BufferedReader(new FileReader(inputFile));
            BufferedWriter writer = new BufferedWriter(new FileWriter(outputFile))
        ) {
            String line;
            while ((line = reader.readLine()) != null) { // 逐行读取
                writer.write(line); // 写入读取的一行
                writer.newLine(); // 写入换行符
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
```