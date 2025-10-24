FROM ubuntu:22.04
RUN apt-get update && apt-get install --upgrade -y curl git python3 python3-pip pipx libpcap0.8 nginx && pipx ensurepath
RUN service nginx restart
EXPOSE 80 3000
RUN ["pip3", "install", "--force", "requests"] 
RUN ["python3", "-m", "pip", "install", "cicflowmeter"]
RUN mkdir -p /out
VOLUME /out
RUN touch /out/logs.csv
ENTRYPOINT [ "cicflowmeter", "-i", "eth0", "-c", "/out/logs.csv" ] 
