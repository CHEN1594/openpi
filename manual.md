先screen -ls一下把没用的先删一下

cd /data/home/zhangjing2/th/openpi/examples/libero
-S libero_server

屏幕内:
conda deactivate
cd /data/home/zhangjing2/th/openpi
source .venv/bin/activate #激活根目录服务器环境
cd examples/libero

uv run --active ../../scripts/serve_policy.py \
  --env LIBERO \
  --port 8001 \
  policy:checkpoint \
  --policy.config pi05_libero \
  --policy.dir /data/home/zhangjing2/th/openpi/pytorch_pi05_leboro

然后ctrl A+D挂起

screen -S libero_client
屏幕内:
conda deactivate
cd /data/home/zhangjing2/th/openpi/examples/libero
source libero/bin/activate
export PYTHONPATH=$PYTHONPATH:$PWD/../../third_party/libero #设置python能指向第三方库
python main.py --args.port 8001 #传入刚才使用的端口
