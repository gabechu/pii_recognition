const resetCallout = () => {
  const emptyString = "";
  document.querySelector("#pii-entities").value = emptyString;
};

document.querySelector("#reset").addEventListener("click", resetCallout);
