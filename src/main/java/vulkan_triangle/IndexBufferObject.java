package vulkan_triangle;

import java.nio.ByteBuffer;

public class IndexBufferObject implements MemCopyable {
    public short[] data;

    public IndexBufferObject(short[] indices) {
        data = indices;
    }

    public void copyTo(ByteBuffer buffer) {
        for (short index : data) {
            buffer.putShort(index);
        }

        buffer.rewind();
    }

    public int getByteLength() {
        return Short.SIZE * data.length;
    }
}
