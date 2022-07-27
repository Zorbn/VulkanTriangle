package vulkan_triangle;

import java.nio.ByteBuffer;

public class VertexBufferObject implements MemCopyable {
    public Vertex[] data;

    public VertexBufferObject(Vertex[] vertices) {
        data = vertices;
    }

    public void copyTo(ByteBuffer buffer) {
        for (Vertex vertex : data) {
            buffer.putFloat(vertex.pos().x());
            buffer.putFloat(vertex.pos().y());

            buffer.putFloat(vertex.color().x());
            buffer.putFloat(vertex.color().y());
            buffer.putFloat(vertex.color().z());

            buffer.putFloat(vertex.texCoords().x());
            buffer.putFloat(vertex.texCoords().y());
        }

        buffer.rewind();
    }

    public int getByteLength() {
        return Vertex.SIZE * data.length;
    }
}
