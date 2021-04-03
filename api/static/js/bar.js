// Set new default font family and font color to mimic Bootstrap's default styling
Chart.defaults.global.defaultFontFamily = '-apple-system,system-ui,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif';
Chart.defaults.global.defaultFontColor = '#292b2c';

// Bar Chart Example
charts = []
for (i=0;i<8;i++){
    var ctx = document.getElementById("myBarChart"+i);

    charts[i] = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: ["Beta", "Alpha", "Theta", "Delta"],
        datasets: [{
          label: "Power",
          backgroundColor: "rgba(2,117,216,1)",
          borderColor: "rgba(2,117,216,1)",
          data: [0, 0, 0, 0,],
        }],
      },
      options: {
        scales: {
          xAxes: [{
            time: {
              unit: 'month'
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
        animation = false
      }

    });
}


function update_bar(){
    $.ajax({
        url: "/getpsd",
        type: "get",
        success: function (response) {
            for (i=0;i<8;i++){
                charts[i].data = {
                    labels: ["Beta", "Alpha", "Theta", "Delta"],
                    datasets: [{
                          label: "Power",
                          backgroundColor: "rgba(2,117,216,1)",
                          borderColor: "rgba(2,117,216,1)",
                          data: [response[i][0], response[i][1], response[i][2], response[i][3]],
                        }],
                      };
                charts.update();
                setTimeout(update_bar,1000);
            }
        },
        error: function (xhr) {
            //Do Something to handle error
        }
    });
};

