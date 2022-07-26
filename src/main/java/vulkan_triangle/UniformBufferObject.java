package vulkan_triangle;

import org.joml.Matrix4f;

import java.nio.ByteBuffer;

public class UniformBufferObject implements MemCopyable {
    public static final int SIZE = 3 * 16 * Float.BYTES;
    // Alignment required by Vulkan spec.
    public static final int MATRIX4F_ALIGNMENT = 4 * Float.BYTES;

    public final Matrix4f model;
    public final Matrix4f view;
    public final Matrix4f proj;

    public UniformBufferObject() {
        model = new Matrix4f();
        view = new Matrix4f();
        proj = new Matrix4f();
    }

    public void copyTo(ByteBuffer buffer) {
        final int mat4Size = 16 * Float.BYTES;

        model.get(0, buffer);
        view.get(AlignmentUtils.align(mat4Size, MATRIX4F_ALIGNMENT), buffer);
        proj.get(AlignmentUtils.align(mat4Size * 2, MATRIX4F_ALIGNMENT), buffer);

        buffer.rewind();
    }

    public int getByteLength() {
        return SIZE;
    }
}
