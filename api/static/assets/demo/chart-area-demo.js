Chart.defaults.global.defaultFontFamily = '-apple-system,system-ui,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif';
Chart.defaults.global.defaultFontColor = '#292b2c';

charts_area = []

label_tab = []
for(i = 0 ; i<1000; i++){
    label_tab.push('')
}

title = ['FT7', 'FT8', 'T7', 'CP1', 'CP2', 'T8', 'O2', 'O1']

for(i =0 ; i<8; i++){
    var ctx = document.getElementById("myAreaChart"+i);
    charts_area[i] = new Chart (ctx, {
        type: 'line',
        data: {
            labels: label_tab,
        },
        options: {
            scales: {
                xAxes: [{
                    time: {
                        unit: 'Hertz'
                    },
                    gridLines: {
                        display: false
                    },
                    ticks: {
                        maxTicksLimit: 6
                    }
                }],
                yAxes: [{
                    ticks: {
                        max : 20000,
                        min : -20000,
                        maxTicksLimit: 5
                    },
                    gridLines: {
                        display: true
                    }
                }],
            },
            legend: {
                display: false
            },
            animation: false,
            title: {
                display: true,
                text: title[i]
            }
        }
    });
}
color = ['red','blue','orange','purple','pink','cyan','magenta','green' ]


function update_area(){
    $.ajax({
        url: "/getraw",
        type: "get",
        success: function (response) {
            if (response != 3){
                for (i=0;i<8;i++){
                    charts_area[i].data = {
                        labels: label_tab,
                        datasets: [{
                              label: "uV/count",
                              data: response[i],
                              fill: false,
                              borderColor: color[i],

                              pointRadius: 0
                            }],
                          };
                    charts_area[i].update();
                }

            }
            setTimeout(update_area,500);
        },
        error: function (xhr) {
            //Do Something to handle error
        }
    });
};

 