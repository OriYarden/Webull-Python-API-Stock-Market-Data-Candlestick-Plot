# Webull-Python-API-Stock-Market-Data-Candlestick-Plot
Plotting candlestick data in Python using Webull's API


To `import` and `login` Webull (plugin your email and account password):

    !pip install webull
    from webull import webull
    wb = webull()
    wb.login('my@email.com', 'password')

To obtain stock data, we'll use the `get_bars` function and specify the stock's symbol, timeframe, and number of samples (i.e. `stock`, `interval`, and `count`) and save it as a `DataFrame` using `pandas`'s `to_csv` function:

    stock_symbol = 'SPY'
    stock_data = wb.get_bars(stock=stock_symbol, interval='m1', count=390, extendTrading=0)

    _date = '2023-09-22'
    file_name = f'/content/drive/My Drive/Colab Notebooks/DATA_FOLDERS/DATA_FRAMES/{stock_symbol}_{_date}.csv'
    import pandas as pd
    stock_data.to_csv(file_name)

where `'m1'` refers to a one-minute timeframe.

We'll then retrieve our saved stock data using `pandas`'s `read_csv` function and parse the `timestamp`s into time and date:

    df = pd.read_csv(file_name)
    
    def parse_date_from_timestamp(timestamp):
        return timestamp[:timestamp.find(' ')]
    
    def parse_time_from_timestamp(timestamp):
        def correct_timestamps(timestamp):
            '''convert timestamp from 24h to 12h'''
            return timestamp if int(timestamp[:2]) <= 12 else '0' + str(int(timestamp[:2]) - 12) + timestamp[2:]
        return correct_timestamps(timestamp[timestamp.find(' ') + 1:timestamp.find('-4:') - 5])
    
    df.date = df.timestamp.map(parse_date_from_timestamp)
    df.timestamp = df.timestamp.map(parse_time_from_timestamp)

We'll `def`ine a `candlestick_plot_function` to group the data into `numpy` `arrays` and then plot the data as candlesticks using a combination of `BoxStyle`, `FancyBboxPatch`, and `Line2D` `from` `matplotlib`:

    import numpy as np
    from matplotlib import pyplot as plt
    from matplotlib.patches import BoxStyle
    from matplotlib.patches import FancyBboxPatch
    from matplotlib.lines import Line2D

    def candlestick_plot_function(fig, ax, df, stock_symbol, candlestick_size_in_minutes=30, wick_linewidth=2.0, fancy_box_padding=0.0005):
        def calc_num_candlesticks(df, candlestick_size_in_minutes):
            return int(df.index.stop / candlestick_size_in_minutes) if df.index.stop / candlestick_size_in_minutes == int(df.index.stop / candlestick_size_in_minutes) else int(df.index.stop / candlestick_size_in_minutes) + 1
    
        def calc_box_width(num_candlesticks):
            return 0.98 - (num_candlesticks / 125)
    
        def remove_prefixed_zero(timestamp):
            return timestamp[:5] if timestamp[0] != '0' else timestamp[1:5]
    
        y = np.zeros((calc_num_candlesticks(df, candlestick_size_in_minutes), 4)).astype(float)
        box_width = calc_box_width(y.shape[0])

        ax.set_facecolor([0, 0, 0.35])
        ax.grid(which='major', axis='both', color=[1, 1, 1], linewidth=0.5, zorder=0)
        for candlestick in range(y.shape[0]):
            indexes_per_candlestick = range(candlestick*candlestick_size_in_minutes, (candlestick + 1)*candlestick_size_in_minutes, 1)
            if indexes_per_candlestick.stop > df.index.stop:
                indexes_per_candlestick = range(candlestick*candlestick_size_in_minutes, df.index.stop, 1)
            data = df.iloc[indexes_per_candlestick]
            y[candlestick, :] = np.array([data.open.iloc[0], np.max(data.high), np.min(data.low), data.close.iloc[-1]])
    
            top_of_box = np.max([y[candlestick, 0], y[candlestick, 3]])
            bottom_of_box = np.min([y[candlestick, 0], y[candlestick, 3]])
            box_color = np.array([0.0, 0.8, 0.6941]) if y[candlestick, 0] < y[candlestick, 3] else np.array([1.0, 0.0, 0.0])
            plot_candlestick = FancyBboxPatch(xy=(candlestick - box_width*0.5, bottom_of_box), width=box_width, height=top_of_box - bottom_of_box, facecolor=box_color, edgecolor=box_color, boxstyle=BoxStyle('round', pad=fancy_box_padding), zorder=2)
    
            ax.add_line(Line2D(xdata=(candlestick, candlestick), ydata=(y[candlestick, 2], bottom_of_box), color=box_color, linewidth=wick_linewidth, antialiased=True, zorder=2))
            ax.add_line(Line2D(xdata=(candlestick, candlestick), ydata=(y[candlestick, 1], bottom_of_box), color=box_color, linewidth=wick_linewidth, antialiased=True, zorder=2))
            ax.add_patch(plot_candlestick)

        ax.set_ylim(np.min(y) - 0.1*(np.max(y) - np.min(y)), np.max(y) + 0.1*(np.max(y) - np.min(y)))
        ax.set_xlim(-0.1*y.shape[0], y.shape[0] + 0.1*y.shape[0])
        ax.set_title(f'''
            {stock_symbol}
            {candlestick_size_in_minutes} Minute Candlesticks''', fontsize=15, fontweight='bold')
        ax.set_yticks([_y for _y in ax.get_yticks()][1:-1])
        ax.set_yticklabels(['{:.2f}'.format(_y) for _y in ax.get_yticks()])
        ax.set_xticks([0, int(y.shape[0]*0.5), y.shape[0] - 1])
        ax.set_xticklabels([remove_prefixed_zero(df.timestamp.values[0]), remove_prefixed_zero(df.timestamp.values[int(df.timestamp.values.shape[0]*0.5)]), remove_prefixed_zero(df.timestamp.values[-1])])
        for axis in ['left', 'right', 'top', 'bottom']:
            ax.spines[axis].set_visible(False) if axis in ['top', 'right'] else ax.spines[axis].set_linewidth(5)
        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            label.set_fontsize(15)
            label.set_fontweight('bold')
    
    fig = plt.figure(figsize=(5, 5))
    ax = plt.subplot(1, 1, 1)
    candlestick_plot_function(fig, ax, df, stock_symbol)
    plt.show()

Which creates a 30-minute candlesticks plot:

![image](https://github.com/OriYarden/Webull-Python-API-Stock-Market-Data-Candlestick-Plot/assets/137197657/3a211deb-93d8-4b6c-a5f0-a155a33020de)

We can also adjust the `candlestick_size_in_minutes` parameter to plot different timeframes; for example 5-minute candlesticks:

    fig = plt.figure(figsize=(5, 5))
    ax = plt.subplot(1, 1, 1)
    candlestick_plot_function(fig, ax, df, stock_symbol, candlestick_size_in_minutes=5, wick_linewidth=0.75)
    plt.show()

![image](https://github.com/OriYarden/Webull-Python-API-Stock-Market-Data-Candlestick-Plot/assets/137197657/07f014a8-a88f-4474-ba24-d507ce629aef)


or 60-minute candlesticks:

    fig = plt.figure(figsize=(5, 5))
    ax = plt.subplot(1, 1, 1)
    candlestick_plot_function(fig, ax, df, stock_symbol, candlestick_size_in_minutes=60)
    plt.show()


![image](https://github.com/OriYarden/Webull-Python-API-Stock-Market-Data-Candlestick-Plot/assets/137197657/1524f528-9fa0-45aa-9935-97df6734d343)

