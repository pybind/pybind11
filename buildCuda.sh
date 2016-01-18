# remove some doubled options & warnings from
#   `python3.5-config --cflags --ldflags --libs`
nvcc -ccbin g++-4.9 \
  -shared -std=c++11 --compiler-options -fPIC \
  -I./include \
  -I/usr/include/python3.5m -I/usr/include/python3.5m -DNDEBUG -g \
  -L/usr/lib/python3.5/config-3.5m-x86_64-linux-gnu -L/usr/lib \
  -Xlinker -export-dynamic \
  -lpython3.5m -lpthread -ldl  -lutil -lrt -lm \
  mandelbrot.cu -o mandelbrot.so
