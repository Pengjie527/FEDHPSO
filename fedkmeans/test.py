import matplotlib.pyplot as plt
from matplotlib import font_manager

font_path = r"C:\Windows\Fonts\SimHei.ttf"
font_manager.fontManager.addfont(font_path)
my_font = font_manager.FontProperties(fname=font_path)

plt.rcParams['font.family'] = my_font.get_name()
plt.rcParams['axes.unicode_minus'] = False

plt.title("✅ 中文字体测试成功！", fontproperties=my_font)
plt.plot([1, 2, 3], [1, 4, 9])
plt.show()
