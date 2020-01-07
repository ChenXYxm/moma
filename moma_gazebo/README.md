# moma_gazebo

Work in progress.

To launch a simulation and some standard trajectory following controllers, run

```
roslaunch mopa_control gazebo.launch
```

To launch a simulation together with MoveIt for motion planning, run

```
roslaunch moma_bringup gazebo_moveit.launch
```

## TODO

- [x] Check if we can move gripper
- [ ] Make second gripper to move the actuated one in gazebo.
- [ ] Fix frames (odom and world are still messed up)
- [x] Set sensible start position in Gazebo simulation