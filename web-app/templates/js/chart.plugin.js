import {GoogleCharts} from 'google-Charts';

GoogleCharts.charts.load('current', {'packages':['corechart']});
GoogleCharts.charts.setOnLoadCallback(drawChart);

function drawChart() {
    let metallic = '{{data[0][2][1][0]}}';
    let metallic_data = parseInt('{{data[0][2][1][1]}}');

    let brown = '{{data[0][2][2][0]}}';
    let brown_data = parseInt('{{data[0][2][2][1]}}');

    let misc = '{{data[0][2][3][0]}}';
    let misc_data = parseInt('{{data[0][2][3][1]}}');

    let green = '{{data[0][2][4][0]}}';
    let green_data = parseInt('{{data[0][2][4][1]}}');

    let gray = '{{data[0][2][5][0]}}';
    let gray_data = parseInt('{{data[0][2][5][1]}}');

    let blue = '{{data[0][2][6][0]}}';
    let blue_data = parseInt('{{data[0][2][6][1]}}');

    let pink = '{{data[0][2][7][0]}}';
    let pink_data = parseInt('{{data[0][2][7][1]}}');

    let red = '{{data[0][2][8][0]}}';
    let red_data = parseInt('{{data[0][2][8][1]}}');

    let violet = '{{data[0][2][9][0]}}';
    let violet_data = parseInt('{{data[0][2][9][1]}}');

    let beige = '{{data[0][2][10][0]}}';
    let beige_data = parseInt('{{data[0][2][10][1]}}');

    let yellow = '{{data[0][2][11][0]}}';
    let yellow_data = parseInt('{{data[0][2][11][1]}}');

    let white = '{{data[0][2][12][0]}}';
    let white_data = parseInt('{{data[0][2][12][1]}}');

    let black = '{{data[0][2][13][0]}}';
    let black_data = parseInt('{{data[0][2][13][1]}}');

    let orange = '{{data[0][2][14][0]}}';
    let orange_data = parseInt('{{data[0][2][14][1]}}');

    let data = GoogleCharts.api.visualization.arrayToDataTable([
        ['{{data[0][2][0][0]}}', '{{data[0][2][0][1]}}'],
        [metallic,     metallic_data],
        [brown,     brown_data],
        [misc,     misc_data],
        [green,     green_data],
        [gray,     gray_data],
        [blue,     blue_data],
        [pink,     pink_data],
        [red,     red_data],
        [violet,     violet_data],
        [beige,     beige_data],
        [yellow,     yellow_data],
        [white,     white_data],
        [black,     black_data],
        [orange,     orange_data]
    ]);
    let options = {
        title: 'Color Distribution',
        colors: ["#dbd9d9", "#996633", "#ccffcc","#009933", "#a7a7a7", "#0000ff", "#ff66ff", "#ff0000", "#993399", "#ffcc66", "#e8f106", "#ffffcc", "#000000", "#ff6600"]
    };

    let chart = new GoogleCharts.visualization.PieChart(document.getElementById('piechart'));

    chart.draw(data, options);
}