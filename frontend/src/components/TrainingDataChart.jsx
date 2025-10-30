
import React from "react";
import { Line } from "react-chartjs-2";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from "chart.js";

// Register chart components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

const TrainingDataChart = ({ data }) => {
  // Prepare data for the chart
  const chartData = {
    labels: data.map((entry) => `Epoch ${entry.epoch}`),
    datasets: [
      {
        label: "Validation Accuracy",
        data: data.map((entry) => entry.val_accuracy),
        fill: false,
        borderColor: "rgba(75,192,192,1)",
        tension: 0.2, // smooth curve
        pointRadius: 4,
      },
    ],
  };

  const options = {
    responsive: true,
    plugins: {
      legend: { position: "top" },
      title: { display: true, text: "Model Validation Accuracy per Epoch" },
    },
    scales: {
      y: {
        beginAtZero: true,
        title: { display: true, text: "Validation Accuracy (%)" },
      },
      x: {
        title: { display: true, text: "Epochs" },
      },
    },
  };

  return (
    <div style={{ width: "80%", margin: "auto" }}>
      <Line data={chartData} options={options} />
    </div>
  );
};

export default TrainingDataChart;
