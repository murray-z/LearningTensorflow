# LearningTensorflow

## CHAPTER 1. Introduction

## CHAPTER 2. Go with the Flow: Up and Running with TensorFlow
- tensorflow 简单示例
    - example2_2.py: MNIST识别，仅将图片像素与权重相乘

## CHAPTER 3. Understanding TensorFlow Basics 
- 计算图(Computation Graph)
    - node: operation(各种函数，运算)
    - edge: flow(数据流，连接各个节点)

- tensorflow 使用步骤
    1. 构建图
    2. 执行图

- 构造图
    - import tensorflow as tf (生成默认图)
    -  tf.<operator> 生成node
    - 示例：
        - 定义常量a, b, c
           ```
              a = tf.constant(5)
              b = tf.constant(2)
              c = tf.constant(3)
           ```
    
        - 定义一些运算
            ```
              d = tf.multiply(a, b)
              e = tf.add(c, d)
              f = tf.sub(d, e)
            ```
            
        - 由a, b, c, d, e, f 构造图如下
        ![](graph/计算图示例.png)  
        
    - tensorflow 运算符
    ![](graph/tensorflow运算符.png)
    
- 构造Session并运行
    - 计算一个值
        ```
        sess = tf.Session()
        outs = sess.run(f)
        sess.close()
        ```
    - Fetches 一次性计算多个值
        ```
        with tf.Session() as sess:
            fetches = [a,b,c,d,e,f]
            outs = sess.run(fetches)
        ```
        
-  数据类型
    - dtype
        - c = tf.constant(4.0, dtype=tf.float64)
    - 转换数据类型
        ```
        x = tf.constant([1,2,3],name='x',dtype=tf.float32)
        print(x.dtype)
        x = tf.cast(x,tf.int64)
        print(x.dtype)
        
        Out:
        <dtype: 'float32'>
        <dtype: 'int64'>
        ```
    - tensorflow 支持数据类型
        ![](graph/tensorflow数据类型.png)

- Tensor           
    - Tensor 数组及其形状
        - get_shape获得Tensor形状大小
            ```
            import numpy as np
            
            c = tf.constant([[1,2,3],
                             [4,5,6]])
                             
            print("Python List input: {}".format(c.get_shape()))
            
            c = tf.constant(np.array([
                                    [[1,2,3],
                                    [4,5,6]],
                                    [[1,1,1],
                                    [2,2,2]]
                                    ]))
            print("3d NumPy array input: {}".format(c.get_shape()))
            
            
            Out:
            Python list input: (2, 3)
            3d NumPy array input: (2, 2, 3)
            ```
    - Tensor 初始化常用方法
        - tf.constant()  # 常数
        - tf.random_normal()  # 正态分布
        - tf.truncated_normal()  # 截断正态分布
            ![](./graph/tensor初始化方法.png)
        
- 矩阵乘法
    - tf.matmul(a, b)
    - 为矩阵新增一维，tf.expand_dims() 
    - 示例
        ```
        # 2x3
        A = tf.constant([[1,2,3],
                           [4,5,6]])
        # 1x3
        x = tf.constant([1,0,1])
        
        # 将一维转为二维 3x1
        x = tf.expand_dims(x,1)
       
        # 2x1
        b = tf.matmul(A,x)
        
        # 交互式session
        sess = tf.InteractiveSession()
        print('matmul result:\n {}'.format(b.eval()))
        sess.close()
        ```
    - 改变tensor形状
        - tf.transpose()
        
- 命名空间 Name Scopes
    - 将一些变量放到同一个命名空间方便管理，类似于nodes group
    - 示例
        ```
        with tf.Graph().as_default():
            c1 = tf.constant(4,dtype=tf.float64,name='c')
            with tf.name_scope("prefix_name"):
                c2 = tf.constant(4,dtype=tf.int32,name='c')
                c3 = tf.constant(4,dtype=tf.float64,name='c')
        
        print(c1.name)
        print(c2.name)
        print(c3.name)
        
        Out:
        c:0
        prefix_name/c:0
        prefix_name/c_1:0
        ```
    
    
    
    
    
    
    
    
    

            
    
    
    

