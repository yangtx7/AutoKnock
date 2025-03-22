import matlab.engine

# 启动 MATLAB 引擎
eng = matlab.engine.start_matlab()

# 运行 MATLAB 表达式
eng.eval("disp('Hello from MATLAB on macOS')", nargout=0)

# MATLAB 函数调用示例
a = eng.linspace(1.0, 10.0, 5)
print("Generated array:", a)

# 关闭引擎
eng.quit()
