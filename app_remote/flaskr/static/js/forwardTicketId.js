/**
 * Forward ticket id to redaction app for a replacement where the ticket id is
 * retrieved from requesting ZAFClient.
 * @param {ZAFClient} zafClient
 * @param {RedactionAppClient} appClient
 */
const forwardTicketId = function (zafClient, appClient) {
  zafClient.get('ticket.id')
    .then(payload => appClient.replaceTicketId(payload))
    .then(response => {
      if (response.status !== 200) {
        console.log(`Request failed. Status code: ${response.status}`)
      }
    })
    .catch(error => console.log(error))
}

export { forwardTicketId }
