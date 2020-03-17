## FeatureCollection与Feature的区别
Feature
> 特征对象必须由一个名字为"geometry"的成员，这个几何成员的值是上面定义的几何对象或者JSON的null值。

> 特征对戏那个必须有一个名字为“properties"的成员，这个属性成员的值是一个对象（任何JSON对象或者JSON的null值）。

> 如果特征是常用的标识符，那么这个标识符应当包含名字为“id”的特征对象成员。


FeatureCollection
> 类型为"FeatureCollection"的GeoJSON对象是特征集合对象。

> 类型为"FeatureCollection"的对象必须由一个名字为"features"的成员。
与“features"相对应的值是一个数组。这个数组中的每个元素都是上面定义的特征对象。




