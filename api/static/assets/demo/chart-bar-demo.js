// Set new default font family and font color to mimic Bootstrap's default styling
Chart.defaults.global.defaultFontFamily = '-apple-system,system-ui,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif';
Chart.defaults.global.defaultFontColor = '#292b2c';

// Bar Chart Example
charts = []
title = ['FT7', 'FT8', 'T7', 'CP1', 'CP2', 'T8', 'O2', 'O1']
color = ['red','blue','orange','purple','pink','cyan','magenta','green' ]

for (i=0;i<8;i++){
    var ctx = document.getElementById("myBarChart"+i);

    charts[i] = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: ["Beta", "Alpha", "Theta", "Delta"],
        datasets: [{
          label: "Power",
          backgroundColor: "rgba(2,117,216,1)",
          borderColor: color[i],
          data: [0, 0, 0, 0,],
        }],
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
              min: 0,
              max: 4000,
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
        animation : false,
        title: {
            display: true,
            text: title[i]
        }
      }

    });
}


function update_bar(){
    $.ajax({
        url: "/getpsd",
        type: "get",
        success: function (response) {
            if (response != 3){
                for (i=0;i<8;i++){
                    charts[i].data = {
                        labels: ["Beta", "Alpha", "Theta", "Delta"],
                        datasets: [{
                              label: "Power",
                              backgroundColor: color[i],
                              borderColor: color[i],
                              data: response[i],
                            }],
                          };
                    charts[i].update('active');
                }

            }
            setTimeout(update_bar,500);
        },
        error: function (xhr) {
            //Do Something to handle error
        }
    });
};

