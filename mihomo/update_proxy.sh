# 配置文件名
CONF="config.yaml"

# 2. 基础端口对齐
# 将 mixed-port 统一设为 7890
sed -i 's/^mixed-port: .*/mixed-port: 7890/' $CONF

# 3. 彻底禁用 TUN (精准匹配 tun: 下方的 enable)
# 找到 tun: 这一行，并把随后的 enable: true 改为 false
sed -i '/tun:/,/enable:/{s/enable: true/enable: false/}' $CONF

# 4. 禁用 IPv6
sed -i 's/^ipv6: .*/ipv6: false/' $CONF

# 5. DNS 适配手术
# 找到 dns: 块，禁用它（容器环境最稳的做法）
sed -i '/dns:/,/enable:/{s/enable: true/enable: false/}' $CONF
# 或者如果你想保留 DNS，改监听端口防止冲突
sed -i "s/listen: .*/listen: '0.0.0.0:1053'/" $CONF
sed -i 's/enhanced-mode: .*/enhanced-mode: redir-host/' $CONF

# 6. 屏蔽外部管理端口的认证（可选，方便本地管理面板连接）
sed -i "s/external-controller: .*/external-controller: '0.0.0.0:9097'/" $CONF

echo "手术完成，正在重启 Mihomo..."

# 7. 重启进程
pkill -9 mihomo
nohup ./mihomo -f $CONF -d . > mihomo.log 2>&1 &

echo "Mihomo 已重启，当前进程 PID: $(pgrep mihomo)"
echo "尝试测试连接..."
sleep 2
curl -I https://www.google.com --connect-timeout 5

# echo 'export http_proxy="http://127.0.0.1:7890"' >> ~/.bashrc
# echo 'export https_proxy="http://127.0.0.1:7890"' >> ~/.bashrc
# echo 'export all_proxy="socks5://127.0.0.1:7890"' >> ~/.bashrc
# source ~/.bashrc
# tail -n 20 mihomo.log
# ps -ef | grep mihomo