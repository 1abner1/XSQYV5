import paramiko

# 具体设置
host = "192.168.2.20"
username = "visbot"
password = "visbot"

# 执行的命令
command = "python3 /root/BigDrone/BigDrone.py"

# 创建 SSH 客户端
client = paramiko.SSHClient()
# 自动添加主机名和密钥到本地的 HostKeys 对象
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

try:
    # 连接到远程主机
    client.connect(hostname=host, username=username, password=password)
    # 打开一个交互式的通道
    print("Exec!")
    stdin, stdout, stderr = client.exec_command(command)
    # 获取执行结果
    output = stdout.read().decode().strip()
    error = stderr.read().decode().strip()
    # 打印执行结果
    if output:
        print(output)
    if error:
        print(error)
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    # 关闭 SSH 连接
    client.close()