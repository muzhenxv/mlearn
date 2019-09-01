transformer模块为特征工程模块，encoder为其基本单元，encoder的作用为进行特征变换。

此文档用于规范encoder类的编写，同时列出各个encoder的参数列表。

# 编写规范

1. 类取名应该清晰易懂， 符合一般规范（各单词首字母大写），以Encoder结尾， 例如`BaseEncoder`, `WOEEncoder`。对于只适用于cont的Encoder类应该以Cont作为类名开头，对于只适用于cate的Encoder类应该以Cate作为类名开头。

2. 类必须包含`__init__`、`fit`和`transformer`方法，transformer模块也只会调用这三个方法

3. fit方法必须形如：

   ```Python
   # 形式1
   def fit(self, df, y=None):
       ···
   
   # 形式2
   def fit(self, df, y):
       ···
   
   # 不需要return
   ```

   

4. transformer方法必须形如：

   ```python
   def transform(self, df):
       # 一系列处理过程
       df_final = df.copy()
       ...
       df_final.columns = df.columns
       return df.final
   
   # return必须是pandas.DataFrame,不能乱序，顺序必须和传入的df一致(最好保持index和传入的df一致，目前会在外部再做一次index对齐处理防止index不一致,最后列名也需要复原出来，因为某些方法会删掉列名)
   ```

   

5. 异常处理和错误抛出规范：不要使用`try...except...`进行异常跳出，目前只需要正常让程序自行报错即可

6. 必须有类注释，形如：

   ```
   """
   用于剔除缺失值严重列，同值严重列，不同值严重cate列（字符串列如果取值太过于分散，则信息量过低）。
   
   适用于cont和cate，支持缺失值, 建议放置在encoder序列第一位次
   
   Parameters
   ----------
   missing_thr: 0.8, 缺失率高于该值的列会被剔除
   
   same_thr: 0.8, 同值率高于该值的列会被剔除
   
   caate_thr: 0.9， 取值分散率高于该值的字符串列会被剔除
   
   Attributes
   ----------
   missing_cols: list, 被剔除的缺失值列
   
   same_cols: list, 被剔除的同值列
   
   cate_cols: list, 被剔除的取值分散字符串列
   
   exclude_cols: list, 被剔除的列名
   """
   ```

7. 完成后必须进行单元测试，自测通过

# BaseEncoder

```
"""
用于剔除缺失值严重列，同值严重列，不同值严重cate列（字符串列如果取值太过于分散，则信息量过低）。

适用于cont和cate，支持缺失值, 建议放置在encoder序列第一位次

Parameters
----------
missing_thr: 0.8, 缺失率高于该值的列会被剔除

same_thr: 0.8, 同值率高于该值的列会被剔除

caate_thr: 0.9， 取值分散率高于该值的字符串列会被剔除

Attributes
----------
missing_cols: list, 被剔除的缺失值列

same_cols: list, 被剔除的同值列

cate_cols: list, 被剔除的取值分散字符串列

exclude_cols: list, 被剔除的列名
"""
```

# NothingEncoder

```
"""
原样返回，不做任何处理。本用于测试，现在transformer支持在encoders序列为空情况下原样返回，此类已无实际用途。

适用于cont和cate，支持缺失值
"""
```

# DropEncoder

```
"""
此类用于返回空df，换言之删除所有字段。

适用于cont和cate， 支持缺失值
"""
```

# ContImupterEncoder

```
"""
此类继承自sklearn.preprocessing.Imputer，fit与基类完全一致，transform方法返回变为pandas.DataFrame。

仅适用于cont， 支持缺失值

Parameters
----------
missing_values : integer or "NaN", optional (default="NaN")
    The placeholder for the missing values. All occurrences of
    `missing_values` will be imputed. For missing values encoded as np.nan,
    use the string value "NaN".

strategy : string, optional (default="mean")
    The imputation strategy.

    - If "mean", then replace missing values using the mean along
      the axis.
    - If "median", then replace missing values using the median along
      the axis.
    - If "most_frequent", then replace missing using the most frequent
      value along the axis.

axis : integer, optional (default=0)
    The axis along which to impute.

    - If `axis=0`, then impute along columns.
    - If `axis=1`, then impute along rows.

verbose : integer, optional (default=0)
    Controls the verbosity of the imputer.

copy : boolean, optional (default=True)
    If True, a copy of X will be created. If False, imputation will
    be done in-place whenever possible. Note that, in the following cases,
    a new copy will always be made, even if `copy=False`:

    - If X is not an array of floating values;
    - If X is sparse and `missing_values=0`;
    - If `axis=0` and X is encoded as a CSR matrix;
    - If `axis=1` and X is encoded as a CSC matrix.

Attributes
----------
enc : 实例化的Imputer对象

enc.statistics_ : array of shape (n_features,)
    The imputation fill value for each feature if axis == 0.

Notes
-----
- When ``axis=0``, columns which only contained missing values at `fit`
  are discarded upon `transform`.
- When ``axis=1``, an exception is raised if there are rows for which it is
  not possible to fill in the missing values (e.g., because they only
  contain missing values).    
"""
```

# ContBinningEncoder

```
"""
将连续型变量转化为离散型

仅适用于cont， 支持缺失值

Parameters
----------
diff_thr: 20, 不同取值数高于该值才进行离散化处理，不然原样返回

binning_method: 'dt'，分箱方法, 可用取值为'dt'， 'qcut'， 'cut'

bins: 10, 分箱数目， 当binning_method='dt'时，该参数失效

**kwargs: 决策树分箱方法使用的决策树参数

"""
```

# WOEEncoder

```
"""
woe变换

适用于cont和cate，但对多取值cont无效，支持缺失值

Parameters
----------
diff_thr: 20， 不同取值数小于等于该值的才进行woe变换，不然原样返回

woe_min: -20， woe的截断最小值

woe_max: 20， woe的截断最大值

nan_thr: 0.01， 对缺失值采用平滑方法计算woe值，nan_thr为平滑参数

"""
```