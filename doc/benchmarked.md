
# env
|os|cpu|core|mem|gpu|显存|
|--|--|--|--|--|--|
|linux-22.04.3|13th Gen Intel(R) Core(TM) i5-13600KF|20|32G|3060-Ti|8G|

# benchmarked

<table>
 <tr>
    <th rowspan="1">模型</th>
    <th rowspan="1">引擎</th>
    <th colspan="1" style="text-align:center">FPS</th>
 </tr>
 <tr>
    <td>person_ball_512*768</td>
    <td>tensorRT</td>
    <td>391</td>
 </tr>

  <tr>
    <td>person_ball</td>
    <td>tensorRT</td>
    <td>424</td>
 </tr>

  <tr>
    <td>person_ball</td>
    <td></td>
    <td></td>
 </tr>

</talbe>

> 帧率 = 1000 / time(解码->推理->编码)
