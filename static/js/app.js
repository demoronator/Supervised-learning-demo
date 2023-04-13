$(document).on({
    ajaxStart: function () {
        $("#loading-indicator").show();
    },
    ajaxStop: function () {
        $("#loading-indicator").hide();
    }
});

function trainModels() {
    // Send AJAX request to train models
    $.ajax({
        type: "POST",
        url: "/train",
        dataType: "text",
        success: function (data) {
            // Show input area

            // Remove "NaN"
            data = data.replace(/NaN/g, '"NaN"');
            valueByFeature = JSON.parse(data);

            // Clear input form
            $("#input-form").empty();

            Object.keys(valueByFeature).forEach(function (featureName) {
                const featureValues = valueByFeature[featureName];
                const div = $("<div></div>");
                if (featureValues.length == 2 && $.isNumeric(featureValues[0]) && $.isNumeric(featureValues[1])) {
                    // Create range input for numeric features
                    div.append($(
                        "<label for='" + featureName + "'>" + featureName + ":</label>"
                    ))
                    div.append($(
                        "<span>" + featureValues[0] + "</span>"
                    ))
                    div.append($(
                        "<input type='range'" +
                        " name='" + featureName + "' title='" + featureName +
                        "' min='" + featureValues[0] + "' max='" + featureValues[1] +
                        "' step='" + ((featureValues[1] - featureValues[0]) / 100) +
                        "' />"
                    ))
                    div.append($(
                        "<span>" + featureValues[1] + "</span>"
                    ))
                } else {
                    // Create select input for categorical features
                    div.append($(
                        "<label for='" + featureName + "'>" + featureName + ":</label>"
                    ))
                    const select = $(
                        "<select id='" + featureName + "' name='" + featureName + "'></select>"
                    )
                    featureValues.forEach(function (value) {
                        select.append("<option value='" + value + "'>" + value + "</option>");
                    })
                    div.append(select)
                }
                $("#input-form").append(div);
            });


            // Add predict button
            $("#input-area").append("<button id='predict-btn' type='button' class='btn' disabled>Make Predictions</button>");
            // Enable second predict button
            $("#predict-btn").prop("disabled", false);
        },
        error: function (error) {
            // Append error message to log area
            $("#log-pre").append("Error training models: " + error.responseText + "\n");
        },
        complete: function () {
            // Scroll to bottom of log area
            $("#log-pre").scrollTop($("#log-pre")[0].scrollHeight);
            // Enable button
            $("#predict-btn").prop("disabled", false);
            $("#test-btn").prop("disabled", false);

            showInputArea();
        }
    });
}

function testModels() {
    // Send AJAX request to test models
    $.ajax({
        type: "POST",
        url: "/test",
        dataType: "text",
        success: function (response) {
            // Append success message to log area
            $("#log-pre").append(response);
        },
        error: function (error) {
            // Append error message to log area
            $("#log-pre").append("Error testing models: " + error.responseText + "\n");
        },
        complete: function () {
            // Scroll to bottom of log area
            $("#log-pre").scrollTop($("#log-pre")[0].scrollHeight);
        }
    });
}

function showInputArea() {
    $(".input-area").show();
}

function makePredictions() {
    // Get input data from form
    const inputData = {};
    $("#input-form :input").each(function () {
        if ($(this).is(":checkbox")) {
            inputData[$(this).attr("name")] = $(this).is(":checked") ? 1 : 0;
        } else {
            inputData[$(this).attr("name")] = $(this).val();
        }
    });

    // Send AJAX request to make predictions
    $.ajax({
        type: "POST",
        url: "/predict",
        contentType: "application/json;charset=UTF-8",
        data: JSON.stringify({ "input_data": inputData }),
        dataType: "text",
        success: function (response) {
            // Append success message to log area
            $("#log-pre").append("Predictions: " + response + "\n");
        },
        error: function (error) {
            // Append error message to log area
            $("#log-pre").append("Error making predictions: " + error.responseText + "\n");
        },
        complete: function () {
            // Scroll to bottom of log area
            $("#log-pre").scrollTop($("#log-pre")[0].scrollHeight);
        }
    });
}

function randomizeFeatures() {
    // Get all input fields in the form
    const inputFields = $("#input-form").find("input, select");

    // Loop through input fields
    inputFields.each(function () {
        // Check if input field is a range input
        if ($(this).is("input[type='range']")) {
            // Generate random value between min and max values
            const minVal = parseFloat($(this).attr("min"));
            const maxVal = parseFloat($(this).attr("max"));
            const randVal = (Math.random() * (maxVal - minVal) + minVal).toFixed(1);

            // Set input field value to random value
            $(this).val(randVal);
        }
        // Check if input field is a select input
        else if ($(this).is("select")) {
            // Get all option values for select input
            const optionValues = $(this).find("option").map(function () {
                return $(this).val();
            }).get();

            // Get random option value and set select input value to it
            const randIndex = Math.floor(Math.random() * optionValues.length);
            const randValue = optionValues[randIndex];
            $(this).val(randValue);
        }
    });
}