# 硬链接和软连接


硬链接和软链接（符号链接）是文件系统中两种不同类型的链接，它们在 Unix-like 操作系统（例如 Linux 和 macOS）中用于引用文件和目录。

## 硬链接（对数据的引用，硬链接以文件为单位，修改会同步到各个硬链接文档中）
硬链接是对文件系统中文件的另一个引用或指针。在技术上讲，硬链接指向文件系统中的 inode（索引节点），inode 包含了文件的元数据和指向文件内容的指针。以下是硬链接的一些关键特征：

- 不同的入口，相同的内容：硬链接为文件内容创建了一个额外的目录入口。所有硬链接看起来像是不同的文件，但实际上它们指向同一个文件内容。
- 引用计数：文件系统为每个 inode 维护一个引用计数。当引用计数达到零时（即没有任何硬链接指向该 inode），文件系统才会删除文件内容。
- 删除行为：删除一个硬链接不会删除文件内容，只要还有其他硬链接指向同一 inode。
- 限制：硬链接不能跨文件系统创建，也不能用于目录。

## 软链接（符号链接）（对入口的引用，软链接以文档或文件夹为单位，修改不会同步到各个软链接文档或文件夹中）
软链接，或称为符号链接，是对另一个文件或目录的引用。它们类似于 Windows 中的快捷方式或 macOS 中的别名。软链接的特点如下：

- 路径引用：软链接包含了目标文件或目录的路径。它们是特殊类型的文件，文件内容实际上是目标路径的文本。
- 独立的文件：软链接在文件系统中是独立的文件实体，拥有自己的 inode 和权限设置。
- 删除和移动：删除或移动目标文件会导致软链接失效，因为它们不再指向有效的路径。
- 灵活性：软链接可以跨文件系统创建，也可以链接到目录。

## 硬链接与软链接的使用场景
- 硬链接：硬链接通常用于备份，或者在不同位置需要同一文件内容的场景，而不想创建内容的实际副本。
- 软链接：软链接用于创建对文件的引用，其中文件可能会移动或改变。它们对于创建可移植链接和引用外部文件系统中的文件很有用。