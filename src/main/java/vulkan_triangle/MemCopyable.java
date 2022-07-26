package vulkan_triangle;

import java.nio.ByteBuffer;

public interface MemCopyable {
    void copyTo(ByteBuffer buffer);
    int getByteLength();
}
