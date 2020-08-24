import { concatStrings } from "../utils.js";
import { detectPiiEntities as comprehendModel } from "../apis/comprehendApis.js";
import { entitiesConcatDelimiter } from "../constants.js";

const spacy = (text) => new Error("Spacy model is not implemented");
const flair = (text) => new Error("Flair model is not implemented");
const comprehend = async (text) => {
  const entityArray = await comprehendModel(text);
  return concatStrings(entityArray, entitiesConcatDelimiter);
};

const models = {
  spacy: spacy,
  flair: flair,
  comprehend: comprehend,
};

export default models;
