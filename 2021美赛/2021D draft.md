## 相似度度量

### 指标的选择

数据指标太多——降维，因果性

两两算距离计算量极大——先聚类，在每一类中可以再算距离

1.用PCA，type of voice相关性不强（原理，数据分析图）不适用于PCA

多指标评价弄出几个综合指标来做聚类（×）综合起来会模糊特征

2.character有相关性（原理上概念上）用PCA

### 聚类

用PCA，算聚类的距离的时候用一下熵权

怎么拼起来？

kmeans用什么算距离，k选什么？

把mode加进来

2.6上午  罗：kmeans，PCA

卢：网络的原理

夏：问题重述，学习gephi

明天下午开始主体部分的论文写作

2.6晚上

network和k均值主体部分写作完成

## 网络

node

directed link

directed node degree=in-degree+out-degree

![image-20210206081059448](C:\Users\86180\AppData\Roaming\Typora\typora-user-images\image-20210206081059448.png)

![image-20210206081118600](C:\Users\86180\AppData\Roaming\Typora\typora-user-images\image-20210206081118600.png)

无向网络一定形成对称矩阵

![image-20210206081247336](C:\Users\86180\AppData\Roaming\Typora\typora-user-images\image-20210206081247336.png)

矩阵中的数表示连接的宽度，weigh

**centrality**：

![image-20210206082215180](C:\Users\86180\AppData\Roaming\Typora\typora-user-images\image-20210206082215180.png)

接近中心度，一个点到图中其他所有点的距离求和，再取倒数

![image-20210206082416470](C:\Users\86180\AppData\Roaming\Typora\typora-user-images\image-20210206082416470.png)

中介中心度

![image-20210206082608734](C:\Users\86180\AppData\Roaming\Typora\typora-user-images\image-20210206082608734.png)

网络质心

质心弥散

加参与者的距离相对于网络质心位置的加权标准差，用来度量网络的覆盖。
传球球员的重要性是不同的，简单的距离标准差不能反映这一点。
加权

Dynamic Network

influencer太多了5000+要有这么多节点吗？

主要考虑gephi的画图效果，邻接矩阵也巨大:sob:

![image-20210206141127207](C:\Users\86180\AppData\Roaming\Typora\typora-user-images\image-20210206141127207.png)

**Linear summation assumption.** When the entropy weight method (EWM) is used to 

analyze the problem, the superposition of indicators is linear addition, rather than exponential or 

other forms of superposition. This is necessary by the requirement of the entropy weight method

选不同的时间点来组网，体现动态1930-2010

1940——1960——1980

1.数据处理：

我们选取1940，1970,2000年的数据来构建网络体现网络随时间的动态变化

每个网络中选择影响的人最多的20位influencer和被影响最多的20位follower作为节点，在他们之间建立从influencer指向follower的有向线段，用“距离（与聚类算法中一致）”的倒数衡量音乐的相似性，并作为连接的权重，在图中用连接线的粗细表示



759491

节点大小——被追随/追随的人越多点的半径越大

这样选最有代表性的人来构网，省略了很多联系，度中心度几乎没有意义

musical influence用距离来衡量，在图上表示成线条的宽度

子网络的分析：

流派内的musical influence表示流派内部的传承

跨流派的musical influence表示流派间的交流

中介中心度衡量在不同流派之间的连接作用

As shown in the figure 2, the nodes in the network represent the players and the number of 

nodes corresponds to the number of players on the field. The size of a node circle depends on 

its degree of nodes, and the larger the degree of nodes, the larger the circle. Moreover, the node 

degree corresponds to the degree centrality of the node, indicating that the importance of the 

player is positively correlated with the number of passes made by the player. Because passing 

is one-way, the link is one-way, meaning passing from one player to another. The width of the 

link indicates the weight, which is the number of passes contained in this link. The more times, 

the wider the link. The position of each node depends on its average position, the average of 

the coordinates of each player passing the ball.

In addition, the court is square and the x-coordinate and the y-coordinate are in the range [0, 

100].

![image-20210206180614755](C:\Users\86180\AppData\Roaming\Typora\typora-user-images\image-20210206180614755.png

动态网络？

评价影响力：多指标综合评价

1940

![image-20210207090124576](C:\Users\86180\AppData\Roaming\Typora\typora-user-images\image-20210207090124576.png)



![image-20210207091901165](C:\Users\86180\AppData\Roaming\Typora\typora-user-images\image-20210207091901165.png)

1970

![image-20210207090652527](C:\Users\86180\AppData\Roaming\Typora\typora-user-images\image-20210207090652527.png)

![image-20210207091154534](C:\Users\86180\AppData\Roaming\Typora\typora-user-images\image-20210207091154534.png)

2000

![image-20210207091736166](C:\Users\86180\AppData\Roaming\Typora\typora-user-images\image-20210207091736166.png)

![image-20210207092241263](C:\Users\86180\AppData\Roaming\Typora\typora-user-images\image-20210207092241263.png)

1970

![image-20210207132557303](C:\Users\86180\AppData\Roaming\Typora\typora-user-images\image-20210207132557303.png)



![image-20210207132709946](C:\Users\86180\AppData\Roaming\Typora\typora-user-images\image-20210207132709946.png)

![image-20210207144104686](C:\Users\86180\AppData\Roaming\Typora\typora-user-images\image-20210207144104686.png)

## 语文建模

主要分析摇滚乐，音乐风格很多，受多方影响

结合一个人从师非常多人

一个人主要受一个人的影响

It is assumed that artists are not influenced by other artists outside of influence_data data set.