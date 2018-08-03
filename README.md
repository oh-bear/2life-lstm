### 算法服务接口文档

#### API
- 接口：`http://server-host/ner`
- 方式：POST
- 参数：content <string>
- 响应：

	```
	{
	    "code": 0,
	    "data": {
			"mood_type": "positive",
			"mood_sub_result": "O",
			"mood_sub_type": {
				"A": 0.2,
				"C": 0.3,
				"E": 0,
				"N": 0.1,
				"O": 0.4
			}
	    },
	    "message": "success"
	}	```

#### 算法接口

- function: `Sentiment_lstm.lstm_predict`
- arguments: `string` <str>
- return:

	```
	{
		"mood_type": "positive",
		"mood_sub_result": "O",
		"mood_sub_type": {
			"A": 0.2,
			"C": 0.3,
			"E": 0,
			"N": 0.1,
			"O": 0.4
		}
	}
	```

---

**问题**

- flask 环境下调用算法接口，第二次之后报错：
	```
	Cannot interpret feed_dict key as Tensor: Tensor Tensor("Placeholder_2:0", shape=(50, 200), dtype=float32) is not an element of this graph.

	解决办法：
	keras.backend.clear_session()
	```

- nginx 502 --> 代码 ImportError(ImportError: libcudnn.so.5: cannot open shared object file: No such file or directory) --> cuda 库软的链接错误（symbol link）。解决方法：

	```
	# 查看 cudnn 库相关的信息，找到错误的链接
	$ sudo ldconfig -v | grep "libcudnn"
	/sbin/ldconfig.real: /usr/local/cuda-8.0/targets/x86_64-linux/lib/libcudnn.so.5 is not a symbolic link

	libcudnn.so.5 -> libcudnn.so.5.1.10

	# 重新建立链接
	sudo ln -sf /usr/local/cuda-8.0/targets/x86_64-linux/lib/libcudnn.so.5.1.10 /usr/local/cuda-8.0/targets/x86_64-linux/lib/libcudnn.so.5
	```
