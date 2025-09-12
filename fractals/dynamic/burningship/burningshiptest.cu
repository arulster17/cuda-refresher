#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

using namespace std;

// ----------------- Settings -----------------
const int WIDTH  = 1600;
const int HEIGHT = 900;

// set up initial positions based on ratio
double ratio = (double) WIDTH   / (double)HEIGHT;
double zoomFactor = 0.05; // Adjust zoom speed
double xmin = -0.75 - ratio; double xmax = -0.75 + ratio;
double ymin = -1; double ymax = 1;
double scrollX = 0.0, scrollY = 0.0;


cudaGraphicsResource* cudaPBO;
GLuint pbo, tex;
GLuint VAO, VBO, EBO, shaderProgram;

// ----------------- Scroll Handler -----------------
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
    scrollX += xoffset;
    scrollY += yoffset;   // accumulate so you can read it later
}

// ----------------- CUDA Kernel -----------------
__global__ void drawGradient(uchar4* pixels, int width, int height, double xmin, double xmax, double ymin, double ymax, double curZoom) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;
    
    double u = xmin + (xmax - xmin) * x / (width-1);
    double v = ymax - (ymax - ymin) * y / (height-1);

    // check if mandlebrot
    
    // adaptive max iterations based on zoom level
    int maxIters = min(2000, (int)(200 + 50 * sqrt(1/curZoom)));
    double zx = 0.0f, zy = 0.0f, zxPrev = 0.0f, zyPrev = 0.0f;
    int iter = 0;
    while (zx*zx + zy*zy < 4.0f && iter < maxIters) {
        zx = abs(zx);
        zy = abs(zy);
        double tmp = zx*zx - zy*zy + u;
        zy = 2.0f * zx * zy + v;
        zx = tmp;
        iter++;
    }
    unsigned char r, g, b, a;
    if (iter == maxIters) {
        r = 0;
        g = 0;
        b = 0;
        a = 255;
    } else {
        double t = logf(iter + 1) / logf(maxIters);
        r = (unsigned char)(128 + 127 * sin(6.2831 * t));
        g = (unsigned char)(128 + 127 * sin(6.2831 * t + 2.094));
        b = (unsigned char)(128 + 127 * sin(6.2831 * t + 4.188));
        a = 255;
    }
    pixels[idx] = make_uchar4(r, g, b, a);
}

// ----------------- Shader Utils -----------------
GLuint createShader(const char* src, GLenum type) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &src, nullptr);
    glCompileShader(shader);
    int success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char log[512];
        glGetShaderInfoLog(shader, 512, nullptr, log);
        std::cerr << "Shader error: " << log << "\n";
    }
    return shader;
}

GLuint createProgram(const char* vsSrc, const char* fsSrc) {
    GLuint vs = createShader(vsSrc, GL_VERTEX_SHADER);
    GLuint fs = createShader(fsSrc, GL_FRAGMENT_SHADER);
    GLuint prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glLinkProgram(prog);
    glDeleteShader(vs);
    glDeleteShader(fs);
    return prog;
}

// ----------------- OpenGL Setup -----------------
void initQuad() {
    float quadVertices[] = {
        // positions   // texcoords
        -1.0f, -1.0f,  0.0f, 0.0f,
         1.0f, -1.0f,  1.0f, 0.0f,
         1.0f,  1.0f,  1.0f, 1.0f,
        -1.0f,  1.0f,  0.0f, 1.0f
    };
    unsigned int indices[] = {0, 1, 2, 0, 2, 3};

    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);

    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), quadVertices, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
    glEnableVertexAttribArray(1);

    glBindVertexArray(0);
}

void initTexturePBO() {
    // PBO
    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, WIDTH * HEIGHT * 4, nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    cudaGraphicsGLRegisterBuffer(&cudaPBO, pbo, cudaGraphicsRegisterFlagsWriteDiscard);

    // Texture
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, WIDTH, HEIGHT, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glBindTexture(GL_TEXTURE_2D, 0);
}

// ----------------- Main -----------------
int main() {
    cout << ratio << endl;
    if (!glfwInit()) {
        std::cerr << "Failed to init GLFW\n";
        return -1;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(WIDTH, HEIGHT, "Burning Ship", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create window\n";
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "Failed to init GLAD\n";
        return -1;
    }

    initQuad();
    initTexturePBO();

    // Minimal shaders
    const char* vsSrc = R"(
        #version 330 core
        layout (location = 0) in vec2 aPos;
        layout (location = 1) in vec2 aTex;
        out vec2 TexCoord;
        void main() {
            TexCoord = aTex;
            gl_Position = vec4(aPos, 0.0, 1.0);
        })";
    const char* fsSrc = R"(
        #version 330 core
        in vec2 TexCoord;
        out vec4 FragColor;
        uniform sampler2D screenTex;
        void main() {
            FragColor = texture(screenTex, TexCoord);
        })";
    shaderProgram = createProgram(vsSrc, fsSrc);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glfwSetScrollCallback(window, scroll_callback);


    // ----------------- Render Loop -----------------
    int cnt = 0; 
    while (!glfwWindowShouldClose(window)) {

        // Handle inputs
        double mx, my, lastX, lastY;
        bool dragging;
        
        // drag
        if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS) {
            double mx, my;
            glfwGetCursorPos(window, &mx, &my);
            my = HEIGHT - my;
            if (!dragging) {
                // just started dragging
                dragging = true;
                lastX = mx;
                lastY = my;
            } else {
                // dragging in progress
                double dx = mx - lastX;
                double dy = my - lastY;

                // convert pixel delta → fractal coordinate delta
                double deltaX = dx / (WIDTH-1) * (xmax - xmin);
                double deltaY = -dy / (HEIGHT-1) * (ymax - ymin); // invert Y

                xmin -= deltaX;
                xmax -= deltaX;
                ymin -= deltaY;
                ymax -= deltaY;

                lastX = mx;
                lastY = my;
            }
        } else {
            dragging = false;
        }

        if (scrollY != 0.0) {
            double mx, my;
            glfwGetCursorPos(window, &mx, &my);

            double mouseX = (mx / (WIDTH-1)) * (xmax - xmin) + xmin;
            double mouseY = ((HEIGHT - my) / (HEIGHT-1)) * (ymax - ymin) + ymin;

            double zoomFactor = (scrollY > 0) ? 0.9 : 1.1;

            xmin += (mouseX - xmin) * (1 - zoomFactor);
            xmax += (mouseX - xmax) * (1 - zoomFactor);
            ymin += (mouseY - ymin) * (1 - zoomFactor);
            ymax += (mouseY - ymax) * (1 - zoomFactor);

            scrollY = 0.0; // consume the event
        }

        // // lerp to zoom in/out
        // if ((glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS)) {
            
        //     // compute mouse position in complex plane
        //     glfwGetCursorPos(window, &mx, &my);
        //     double mouseX = (mx / (WIDTH-1)) * (xmax - xmin) + xmin;
        //     double mouseY = ((HEIGHT - my) / (HEIGHT-1)) * (ymax - ymin) + ymin;

        //     xmin += (mouseX - xmin) * zoomFactor;
        //     xmax += (mouseX - xmax) * zoomFactor;
        //     ymin += (mouseY - ymin) * zoomFactor;
        //     ymax += (mouseY - ymax) * zoomFactor;
        // }
        // if ((glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS)) {
        //     // compute mouse position in complex plane
        //     glfwGetCursorPos(window, &mx, &my);
        //     double mouseX = (mx / (WIDTH-1)) * (xmax - xmin) + xmin;
        //     double mouseY = ((HEIGHT - my) / (HEIGHT-1)) * (ymax - ymin) + ymin;
        //     //double mouseX = -0.75;
        //     //double mouseY = 0;
        //     xmin -= (mouseX - xmin) * zoomFactor;
        //     xmax -= (mouseX - xmax) * zoomFactor;
        //     ymin -= (mouseY - ymin) * zoomFactor;
        //     ymax -= (mouseY - ymax) * zoomFactor;
        // }

        double zoomFactor = 0.05; // Adjust zoom speed
        

        //cout << "Mouse: " << xpos << ", " << ypos << "\n";
        cnt++;
        //cout << "Frame: " << cnt << "\n";







        // Map CUDA buffer
        cudaGraphicsMapResources(1, &cudaPBO, 0);
        uchar4* devPtr;
        size_t size;
        cudaGraphicsResourceGetMappedPointer((void**)&devPtr, &size, cudaPBO);

        // Run kernel
        dim3 block(16, 16);
        dim3 grid((WIDTH + 15) / 16, (HEIGHT + 15) / 16);
        drawGradient<<<grid, block>>>(devPtr, WIDTH, HEIGHT, xmin, xmax, ymin, ymax, (xmax - xmin) / (2.0 * ratio));

        cudaGraphicsUnmapResources(1, &cudaPBO, 0);

        // Copy PBO → Texture
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
        glBindTexture(GL_TEXTURE_2D, tex);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, WIDTH, HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

        // Draw fullscreen quad
        glClear(GL_COLOR_BUFFER_BIT);
        glUseProgram(shaderProgram);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, tex);
        glUniform1i(glGetUniformLocation(shaderProgram, "screenTex"), 0);
        glBindVertexArray(VAO);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // Cleanup
    cudaGraphicsUnregisterResource(cudaPBO);
    glDeleteBuffers(1, &pbo);
    glDeleteTextures(1, &tex);
    glfwTerminate();
    return 0;
}
