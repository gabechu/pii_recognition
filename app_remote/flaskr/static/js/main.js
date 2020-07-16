import { forwardTicketId } from './forwardTicketId.js'
import { RedactionAppClient } from './redactionAppClient.js'

const zafClient = ZAFClient.init() // takes some time to initialise
const appClient = new RedactionAppClient(window.origin)
forwardTicketId(zafClient, appClient)
