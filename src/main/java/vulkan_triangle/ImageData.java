package vulkan_triangle;

import java.nio.ByteBuffer;

public class ImageData implements MemCopyable {
    private final ByteBuffer data;
    private final int width, height, channels;

    public ImageData(ByteBuffer pixels, int width, int height, int channels) {
        data = pixels;
        this.width = width;
        this.height = height;
        this.channels = channels;
    }

    @Override
    public void copyTo(ByteBuffer buffer) {
        data.limit(getByteLength());
        buffer.put(data);
        data.limit(data.capacity()).rewind();
    }

    @Override
    public int getByteLength() {
        return width * height * channels;
    }
}
