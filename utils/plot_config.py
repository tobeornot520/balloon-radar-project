from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager


FONT_CANDIDATES = [
    Path(
        "/usr/share/fonts/opentype/noto/"
        "NotoSansCJK-Regular.ttc"
    ),
    Path(
        "/usr/share/fonts/truetype/wqy/"
        "wqy-zenhei.ttc"
    ),
]


def configure_chinese_font():
    """
    为Matplotlib统一配置中文字体。

    返回FontProperties，必要时也可以显式传给标题。
    """

    for font_path in FONT_CANDIDATES:
        if not font_path.exists():
            continue

        try:
            font_manager.fontManager.addfont(
                str(font_path)
            )
        except Exception:
            # 某些Matplotlib版本对TTC字体重复注册会报错，
            # 但不影响继续使用字体文件。
            pass

        font_property = (
            font_manager.FontProperties(
                fname=str(font_path)
            )
        )

        font_name = (
            font_property.get_name()
        )

        plt.rcParams["font.family"] = (
            "sans-serif"
        )

        plt.rcParams["font.sans-serif"] = [
            font_name,
            "DejaVu Sans",
        ]

        plt.rcParams[
            "axes.unicode_minus"
        ] = False

        print(
            f"当前中文绘图字体：{font_name}"
        )

        return font_property

    plt.rcParams[
        "axes.unicode_minus"
    ] = False

    print(
        "警告：没有找到中文字体，"
        "绘图中文字可能显示异常。"
    )

    return None