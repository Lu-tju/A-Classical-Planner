0 依赖：
```
pip install numpy casadi open3d scipy
```
1 启动Control Simulator(同YOPO)

2 启动规划
```
python planner_node.py 
```
3 发布地图可视化
```
python pub_map.py
```
4 rviz 添加地图、轨迹等进行可视化

```
rviz -d yopo.rviz
```

![示例图片](vis.gif)
