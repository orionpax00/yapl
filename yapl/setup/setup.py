import os
import logging
import subprocess
from pyngrok import ngrok


try:
    from google.colab import drive

    colab_env = True
except ImportError:
    colab_env = False


EXTENSIONS = ["ms-python.python", "jithurjacob.nbpreviewer"]


class Setup():
    def __init__(self, port=10000, password=None, mount_drive=False, logdir='./logs', workdir="/kaggle/working"):
        self.dir_path = logdir
        self.workdir = workdir
        self._mount = mount_drive
        self.port = port
        self.password = password
        self._install_code()
        self._install_extensions()
        self._start_server()
        self._run_code()
#         self._tensorBoard()
        self._lab()

    def _install_code(self):
        subprocess.run(
            ["wget", "https://code-server.dev/install.sh"], stdout=subprocess.PIPE
        )
        subprocess.run(["sh", "install.sh"], stdout=subprocess.PIPE)

    def _install_extensions(self):
        for ext in EXTENSIONS:
            subprocess.run(["code-server", "--install-extension", f"{ext}"])

    def _start_server(self):
        active_tunnels = ngrok.get_tunnels()
        for tunnel in active_tunnels:
            public_url = tunnel.public_url
            ngrok.disconnect(public_url)
        url_code = ngrok.connect(port=self.port, options={"bind_tls": True}) ## only two tunnels are available at a time
#         url_board = ngrok.connect(port=self.port+1, options={"bind_tls": True})
        url_lab = ngrok.connect(port=self.port + 2, options={"bind_tls": True})
        print(f"Code Server can be accessed on: {url_code}")
#         print(f"Tensorboard can be accessed on: {url_board}")
        print(f"Lab can be accessed on: {url_lab}")

    def _run_code(self):
        os.system(f"fuser -n tcp -k {self.port}")
        if self._mount and colab_env:
            drive.mount("/content/drive")
        if self.password:
            code_cmd = f"PASSWORD={self.password} code-server {self.workdir} --port {self.port} --disable-telemetry"
        else:
            code_cmd = f"code-server {self.workdir} --port {self.port} --auth none --disable-telemetry"
        subprocess.Popen(
            [code_cmd],
            shell=True,
            stdout=subprocess.PIPE,
            bufsize=1,
            universal_newlines=True,
        ) 

    def _tensorBoard(self):
        board_cmd = f"tensorboard --logdir={self.dir_path} --port={self.port+1}" 
        subprocess.Popen(
            [board_cmd],
            shell=True,
            stdout=subprocess.PIPE,
            bufsize=1,
            universal_newlines=True,
        )

    def _lab(self):
        lab_cmd = f"jupyter lab {self.workdir} --ip 0.0.0.0 --port {self.port + 2} --no-browser --allow-root" 
        subprocess.Popen(
            [lab_cmd],
            shell=True,
            stdout=subprocess.PIPE,
            bufsize=1,
            universal_newlines=True,
        )

    def killall(self):
        active_tunnels = ngrok.get_tunnels()
        for tunnel in active_tunnels:
            public_url = tunnel.public_url
            ngrok.disconnect(public_url)
