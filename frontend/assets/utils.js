const stripHtml = (text) => {
  text = text.replace(/<br\s*\/?>/gi, " ");
  let dom = document.createElement("div");
  dom.innerHTML = text;
  return dom.textContent || dom.innerText;
};

const concatStrings = (strings, delimiter) => {
  return strings.join(delimiter);
};

const removeWhitespace = (string) => {
  return string.trim();
};

export { stripHtml, concatStrings, removeWhitespace };
