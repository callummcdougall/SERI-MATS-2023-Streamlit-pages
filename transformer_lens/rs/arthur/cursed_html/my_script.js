function createNetworkGraph(numLeft, numRight, values, colorList, labelsLeft, labelsRight, width, height, title) {
    const nodes = [];
    const links = [];
    let maxRowLength = Math.max(numLeft, numRight);
  
    for (let i = 0; i < numLeft; i++) {
        nodes.push({ id: i, x: 80, y: 45 + (i * (height - 100) / (numLeft - 1)) });
      }
    
      for (let i = 0; i < numRight; i++) {
        nodes.push({ id: i + numLeft, x: 520, y: 40 + (i * (height - 100) / (numRight - 1)) });
      }
    
      let colorIndex = 0;
      for (let j = 0; j < numRight; j++) {
        for (let i = 0; i < numLeft; i++) {
          links.push({ source: i, target: j + numLeft, opacity: values[j][i], color: colorList[colorIndex] ? 'red' : 'blue' });
          colorIndex++;
        }
      }
  
    return `
      <!DOCTYPE html>
      <html lang="en">
      <head>
      <meta charset="UTF-8">
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <script src="https://d3js.org/d3.v7.min.js"></script>
      <style>
        .node {
          fill: black;
        }
        .link {
          fill: none;
        }
        .label {
          font-size: 18px;  /* Increased font size */
        }
      </style>
      </head>
      <body>
      <script>
        const svg = d3.select("body").append("svg")
                      .attr("width", ${width})
                      .attr("height", ${height});
  
        svg.append("text")
           .attr("x", ${width / 2})
           .attr("y", 20)
  
        const nodes = ${JSON.stringify(nodes)};
        const links = ${JSON.stringify(links)};
  
        svg.selectAll(".link")
          .data(links)
          .enter().append("path")
          .attr("class", "link")
          .attr("d", d => {
            const dx = nodes[d.target].x - nodes[d.source].x;
            const dy = nodes[d.target].y - nodes[d.source].y;
            const mx = nodes[d.source].x + dx/2;
            const my = nodes[d.source].y + dy/2;
            return \`M\${nodes[d.source].x},\${nodes[d.source].y}C\${mx},\${nodes[d.source].y} \${mx},\${nodes[d.target].y} \${nodes[d.target].x},\${nodes[d.target].y}\`;
          })
          .style("opacity", d => d.opacity === 0.5 ? 0 : 1)
          .style("stroke", d => d.color)
          .style("stroke-width", d => 3 * (d.opacity ** 2));
  
        svg.selectAll(".node")
          .data(nodes)
          .enter().append("circle")
          .attr("class", "node")
          .attr("r", 5)
          .attr("cx", d => d.x)
          .attr("cy", d => d.y);
  
        svg.selectAll(".labelLeft")
          .data(${JSON.stringify(labelsLeft)})
          .enter().append("text")
          .attr("class", "label")
          .attr("x", d => 10)
          .attr("y", (d, i) => 50 + (i * (height - 100) / (numLeft - 1)))
          .text(d => d);
  
        svg.selectAll(".labelRight")
          .data(${JSON.stringify(labelsRight)})
          .enter().append("text")
          .attr("class", "label")
          .attr("x", d => ${width} - 60)  /* Adjusted x position */
          .attr("y", (d, i) => 45 + (i * (height - 100) / (numRight - 1)))
          .text(d => d);
        svg.append("text")
          .attr("x", ${width / 2})
          .attr("y", 20)
          .attr("class", "title")
          .style("font-family", "Arial")  // Change to your desired font
          .style("font-size", "24px")  // Change font size
          .text("${title}");
  
      </script>
      </body>
      </html>
    `;
  }
  