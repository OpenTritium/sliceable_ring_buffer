# 为什么容量始终虚大

因为需要计算 T 和 内存分配粒度（在windows 上使用 dwAllocationGranularity，在 unix/linux 上使用pagsize）的 LCM，才能在缓冲区饱满的时候确保环绕正确

在 Windows 上 的内存分配粒度始终偏粗，是因为 dwAllocationGranularity 几乎总是 64KiB（其他平台几乎是 4KiB）这个设计的历史原因是为了兼容一些早期 RISC 处理器，简化地址计算
